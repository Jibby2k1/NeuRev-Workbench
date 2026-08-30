# NeuRev Learned Whitening and ICA Utility Program

## Codex-oriented sequential experiment and decision handoff

**Repository:** `Jibby2k1/NeuRev-Workbench`
**Dataset:** `Spon Ca Burst / 3 hindbrain to tail 488 20ms`
**Status:** implementation and staged-experiment plan; this document does not itself authorize an unattended long GPU run
**Primary scope:** determine whether learned whitening/operator selection is useful, whether its scale should be global or adaptive, and whether ICA contributes anything beyond the matched whitened representation
**Current-data limitation:** all scientific conclusions are within-video conclusions until additional recordings or fish become available

---

## 1. Executive directive

Implement a new, additive NeuRev experiment program that answers the following questions in order:

1. Can the current evaluation contract be reproduced exactly?
2. Does any well-conditioned spatial, temporal, or separable spatiotemporal whitening operator improve the current Raw Direct representation?
3. Can a small learned global mixture outperform a leakage-safe selection of the best fixed operator from the same bank?
4. Can a continuous whitening scale replace the discrete bank without losing performance?
5. Is there evidence that the appropriate operator must vary across time or space?
6. After controlling for whitening, rank, downstream evidence construction, candidate budget, and random rotation, does ICA provide incremental biological utility?

Do **not** implement a large end-to-end architecture first. Do **not** launch all stages automatically. Every stage must write a machine-readable decision and the next stage must refuse to run unless its prerequisite decision permits it.

The research program is intentionally sequential:

```text
freeze evaluator
    ↓
operator numerical screen
    ↓
fixed-operator response screen
    ↓
global learned mixture
    ↓
continuous learned scale, only if earned
    ↓
adaptive selection, only if heterogeneity is demonstrated
    ↓
matched ICA marginal-utility test
    ↓
stop and summarize
```

A failed idea may receive **one bounded diagnostic rescue** when the failure appears attributable to conditioning, score calibration, or optimization. A second failure terminates that branch.

---

## 2. Repository facts that constrain this program

Before changing code, Codex must read:

- `AGENTS.md`
- `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`
- `docs/CODEBASE_NAVIGATION.md`
- `docs/workflows/spon_ca_burst_representation_benchmark.md`
- `docs/workflows/spon_ca_burst_learnable_contrast.md`
- `docs/workflows/spon_ca_burst_multiscale_information.md`
- `docs/workflows/spon_ca_burst_pairwise_separation.md`
- `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md`

Relevant existing implementation routes include:

- `neurobench/metrics/sparse_detection.py`
- `neurobench/experiments/representation_benchmark/`
- `neurobench/experiments/learnable_contrast/`
- `neurobench/experiments/pairwise_separation/`
- `neurobench/algorithms/pairwise_separation.py`
- `neurobench/algorithms/multiscale_subspace.py`
- `neurobench/cli/experiment.py`

Do not extend legacy top-level `core/`, `evaluation/`, or `reporting/` code when a maintained `neurobench/` route exists.

### 2.1 Existing evidence that must shape the design

The current repository already shows:

- Raw Direct is difficult to beat.
- At the fixed 58-candidate-per-burst budget, Raw Direct recovered 52 of 79 known occurrences, with reported mean burst recall approximately `0.6572`.
- Under the quiet-threshold operating point, Raw Direct recovered 49 of 79 known occurrences, with exact mean burst recall `0.6056159420289855` and 232 event candidates.
- Amplitude PCA rank 8 produced a provisional fixed-budget gain of two known occurrences.
- Spatial FastICA rank 16 was stable but did not beat Raw Direct at fixed budget.
- Rank-64 ICA failed to converge reliably and must not be repeated in the initial ICA program.
- The prior learnable-contrast program did not beat the matched direct baseline.
- Objective scaling materially affected optimization, while initialization jitter was secondary.
- A raw-direct tuning program tied the frozen direct baseline, indicating that merely adding a few trainable degrees of freedom is not sufficient.
- Spatiotemporal preprocessing can be destructive; the prior Kalman/Gaussian/whitened learnable-contrast lane reduced direct recall substantially.
- Unmatched candidates are unknown because the annotations are sparse. They are not valid false positives.

The new program must therefore:

- include Raw Direct as an explicit residual/skip option;
- avoid high-rank ICA;
- separate numerical failure from scientific failure;
- compare learned selection with a fair fixed-selection baseline;
- use fixed candidate budgets for the primary metric;
- prohibit claims of precision;
- keep the first learned models very small;
- treat full joint spatiotemporal whitening as a guarded diagnostic, not the default.

---

## 3. Fixed data and annotation contract

Use the existing local paths from `AGENTS.md`:

```text
Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif
Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.xlsx
Inputs/Spon Ca Burst/labels/labels_normalized.tsv
Inputs/Spon Ca Burst/labels/label_summary.json
Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/
  spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy
```

Current fixed facts:

- video shape: `[2359, 340, 573]`;
- frame rate: 50 Hz;
- frame period: 20 ms;
- quiet interval: UI frames 1800–1899, inclusive;
- scored interval: UI frames 1900–2359, inclusive;
- four labeled burst windows;
- 79 labeled burst occurrences;
- 27 unique coordinates;
- coordinates: `x = column`, `y = row`;
- UI frames: one-based and inclusive;
- NumPy intervals: zero-based and half-open.

Every new label-driven preflight must generate and validate a projection overlay. Any disagreement in coordinate or frame conversion is a hard stop.

### 3.1 Within-video interpretation

This program may establish:

- held-out-burst performance within the current video;
- hidden-ROI performance within the current video;
- numerical stability;
- operator sensitivity;
- evidence that one statistical assumption is more useful than another on this recording.

It may **not** establish:

- cross-fish generalization;
- cross-recording generalization;
- robustness to a new microscope or acquisition condition;
- population-level biological validity.

Reports must state this limitation explicitly.

---

## 4. Canonical evaluation contract

### 4.1 Primary metric

Use **held-out-burst macro known-positive recall at 58 candidates per burst**, abbreviated:

```text
Macro-KPR@58
```

For burst \(b\), let:

- \(n_b\) be the number of labeled occurrences;
- \(m_b\) be the number of one-to-one matched candidates among the top 58 ranked candidates;
- matching radius be 6 pixels;
- NMS distance be 6 pixels.

Then:

\[
R_b = \frac{m_b}{n_b},
\qquad
\operatorname{MacroKPR@58}
=
\frac{1}{4}\sum_{b=1}^{4} R_b.
\]

For stochastic methods, aggregate seeds in this order:

1. compute \(R_{b,s}\) for each burst \(b\) and seed \(s\);
2. take the median over seeds for each burst;
3. average the four burst medians.

This prevents a burst with more labels from dominating the primary score and prevents a single favorable seed from dominating a stochastic method.

### 4.2 Essential companion values

Always retain:

- `Macro-KPR@58`;
- pooled known-positive recall at 58 candidates;
- total known matches out of 79;
- four-element per-burst match and recall vectors;
- delta versus Raw Direct;
- delta versus the leakage-safe fixed-selection comparator.

The pooled value is descriptive, not primary:

\[
\operatorname{PooledKPR@58}
=
\frac{\sum_b m_b}{79}.
\]

### 4.3 Fixed evidence construction

Unless a stage explicitly tests the evidence construction itself, freeze:

- temporal pooling: `lme0.25`;
- NMS distance: 6 pixels;
- one-to-one matching radius: 6 pixels;
- candidate budget: 58 per burst;
- deterministic tie order: descending score, then ascending `y`, then ascending `x`;
- quiet interval and event intervals;
- label parsing and coordinate conversion;
- candidate matching implementation.

Do not allow each method to tune its own pooling temperature, NMS radius, matching radius, or candidate budget during the primary comparison.

### 4.4 Secondary operating-point metric

Retain the existing quiet-calibrated result as a secondary diagnostic:

```text
Macro-KPR@Q1
```

where the threshold is calibrated from quiet pseudo-burst maps at one quiet peak per map.

Also retain:

- total event candidates at the Q1 threshold;
- total known matches at Q1.

This metric is useful for diagnosing selectivity but does not identify precision.

### 4.5 Trace-preservation guardrail

For each held-out labeled occurrence:

1. extract the Raw Direct positive-residual trace within a 2-pixel disk;
2. extract the method’s pre-temporal-pooling evidence trace within the same disk;
3. robust-standardize both using the quiet interval at that location;
4. compute zero-lag Spearman correlation over the labeled burst window extended by two frames on each side.

Define:

\[
\operatorname{TracePreserve}
=
\operatorname{median}_i
\rho_{\mathrm{Spearman}}
\left(
r_i(t), s_i(t)
\right).
\]

This is a diagnostic, not an optimization target.

Trigger a preservation investigation when:

- `TracePreserve < 0.50`; or
- it is more than `0.15` below the matched comparator; or
- visual review shows systematic timing inversion, spatial spreading, or event suppression.

Do not automatically reject a temporal-derivative-like representation solely because its trace correlation is low. Instead, classify it correctly and inspect whether its biological timing remains usable.

### 4.6 Numerical-health guardrail

Every operator or trained model must report:

- finite-output fraction;
- maximum regularized covariance condition number;
- effective-rank fraction;
- eigendecomposition or optimizer convergence;
- resource-limit status.

Default validity conditions:

```text
finite_output_fraction == 1.0
max_regularized_condition_number <= 1e6
effective_rank_fraction >= 0.25
no NaN or Inf parameters
no resource-limit violation
```

These are engineering defaults. A stage may revise one threshold once, but only through a recorded diagnostic rescue with a written reason.

---

## 5. Data splitting and leakage control

### 5.1 Outer evaluation: leave one burst out

Use four outer folds.

For outer fold \(b\):

- burst \(b\) is held out from supervised learning, early stopping, architecture selection, operator selection, and threshold selection;
- the remaining three bursts are outer-training bursts;
- the quiet interval remains available for label-free normalization and null calibration.

### 5.2 Inner model selection

Within each outer fold, rotate one of the three outer-training bursts as inner validation and use the other two for gradient updates.

For each candidate training configuration:

1. fit on two bursts;
2. evaluate on the third;
3. rotate through all three inner validation bursts;
4. select the configuration using mean inner `KPR@58`;
5. break ties by lower validation ranking loss;
6. break remaining ties by lower model complexity;
7. choose the refit epoch as the median best epoch from the three inner folds;
8. refit on all three outer-training bursts for that fixed epoch count;
9. evaluate once on the outer held-out burst.

The outer held-out burst must never be queried during training.

### 5.3 ROI-identity guardrail

The same ROI coordinate can recur across bursts. Although the initial models are translation equivariant and must not receive absolute coordinates, a final learned method must also undergo a deterministic hidden-identity test:

- partition the 27 unique coordinates into four approximately balanced identity folds, stratified by recurrence count;
- hide all labels for held-out identities during training;
- leave their event pixels unknown, not negative;
- evaluate their recovery under the same frozen candidate formation;
- report hidden-identity `Recall@20` and `Recall@58`.

This is a confirmation guardrail, not the primary metric. It is mandatory for adaptive models and strongly recommended for the global mixture finalist.

### 5.4 Prohibited leakage

Do not:

- select QMC points using outer held-out labels;
- select the operator bank using label performance;
- use held-out bursts for early stopping;
- use hidden identities as quiet negatives;
- tune candidate budget on labels;
- use absolute `x`, `y`, frame index, or burst ID as gating-network inputs;
- use the final audit to revise the model.

---

## 6. Fair comparators

Every learned method must be compared with all applicable controls below.

### 6.1 Raw Direct

The original quiet-median positive residual and frozen detector contract.

This is the biological-preservation floor and exact reproducibility check.

### 6.2 Fixed-Select

From the same eligible operator bank used by the learned mixture, select one fixed operator using only inner-training and inner-validation bursts. Apply that selected operator to the outer held-out burst.

This is the primary comparator for the claim:

> learning the operator combination is more useful than selecting a fixed preprocessing configuration.

### 6.3 Fixed Oracle

For diagnosis only, report the best operator after observing all four bursts.

Never use this as a deployable comparator and never use it for a promotion claim.

### 6.4 Uniform Mix

Average the same standardized bank with fixed equal weights.

This determines whether any gain comes from learning or merely from ensembling.

### 6.5 Random Simplex Mix

Use a deterministic set of random simplex weights, frozen before label evaluation.

This controls for generic mixing capacity.

### 6.6 ICA-matched controls

During the ICA stage, include:

- whitened representation with identity/no ICA rotation;
- matched PCA/no-rotation evidence;
- random orthogonal rotations;
- ICA rotations.

ICA is useful only if it beats the matched non-ICA and random-rotation controls.

---

## 7. Scientific-audit and minimal-storage policy

The repository’s scientific-audit output standard is default-on. The user has explicitly requested essential metrics only for the screening program.

For Stages 0–4 screening and non-finalist ICA cells, resolve:

```json
{
  "scientific_audit": {
    "enabled": false,
    "opt_out_reason": "User explicitly requested essential-metrics-only sequential screening on 2026-08-19. Full scientific audit is mandatory for promoted finalists."
  }
}
```

This exemption applies only to screening.

A promoted final method must run with the full audit enabled and must produce the required expert-only, model-only, and matched-comparison evidence.

Do not create full videos, TIFF stacks, feature-map galleries, or per-epoch plots for every screen cell.

---

## 8. Proposed repository layout

Add the following, preserving existing workflows:

```text
docs/developer/
  LEARNED_WHITENING_ICA_SEQUENTIAL_EXPERIMENT_HANDOFF.md

docs/workflows/
  spon_ca_burst_learned_operator_selection.md

neurobench/algorithms/
  local_whitening.py
  learned_operator_mixture.py

neurobench/experiments/learned_operator_selection/
  __init__.py
  __main__.py
  config.py
  data.py
  design.py
  operators.py
  training.py
  evaluation.py
  decisions.py
  preflight.py
  runner.py
  report.py
  ica_ablation.py

examples/
  spon_ca_burst_learned_operator_selection.example.json

tests/
  test_local_whitening.py
  test_learned_operator_design.py
  test_learned_operator_evaluation.py
  test_learned_operator_decisions.py
  test_learned_operator_experiment.py
```

Reusable mathematics belongs in `neurobench/algorithms/`. Dataset-specific orchestration belongs in `neurobench/experiments/learned_operator_selection/`.

The CLI group should remain thin and lazy-load numerical dependencies.

---

## 9. CLI contract

Implement:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learned-operator preflight \
  --config examples/spon_ca_burst_learned_operator_selection.example.json \
  --artifact-dir Outputs/LearnedOperatorSelection/<id>_preflight
```

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learned-operator status \
  --program-dir Outputs/LearnedOperatorSelection/<id>
```

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learned-operator run-stage \
  --config examples/spon_ca_burst_learned_operator_selection.example.json \
  --preflight-dir Outputs/LearnedOperatorSelection/<id>_preflight \
  --stage S0_BASELINE
```

Later stages use the same command with one of:

```text
S1_OPERATOR_SCREEN
S2_GLOBAL_MIXTURE
S3_CONTINUOUS_SCALE
S4_ADAPTIVE_SELECTION
S5_ICA_UTILITY
S6_FINAL_AUDIT
```

Optional bounded controls:

```text
--prefix-size 8|16|32|64
--outer-fold 1|2|3|4
--seed 7
--smoke
--diagnostic-rescue
```

The runner must refuse:

- output collisions;
- a stage whose prerequisite decision is not `advance`;
- `--diagnostic-rescue` when a rescue has already been consumed;
- prefix sizes that are not declared in the master design;
- a full run without a matching ready preflight;
- a final result with scientific audit disabled;
- any attempt to overwrite a completed root.

There must be no `--run-all` command.

---

## 10. Program state machine

Create:

```text
program_state.json
```

with one record per stage.

Allowed stage states:

```text
not_started
preflight_ready
running
completed
failed_operationally
advance
advance_with_caution
diagnostic_rescue
stop_branch
stop_program
awaiting_human_review
```

Every completed stage writes:

```text
stage_summary.json
decision.json
runs.tsv
resolved_config.json
```

`decision.json` must contain:

```json
{
  "stage": "S2_GLOBAL_MIXTURE",
  "decision": "advance",
  "primary_metric": "Macro-KPR@58",
  "comparator": "Fixed-Select",
  "delta_primary": 0.0,
  "pooled_match_delta": 0,
  "burst_delta_vector": [0.0, 0.0, 0.0, 0.0],
  "numerical_health": "pass",
  "preservation_status": "pass_or_review",
  "reason_codes": [],
  "rescue_consumed": false,
  "next_allowed_stages": ["S3_CONTINUOUS_SCALE", "S5_ICA_UTILITY"]
}
```

Do not infer advancement from the existence of files. Read `decision.json`.

---

## 11. Mathematical operator contract

### 11.1 Base input

Let \(X_t(x,y)\) be the raw video.

Fit a quiet baseline \(B(x,y)\) and robust quiet scale \(S(x,y)\) from UI frames 1800–1899 only:

\[
R_t(x,y)
=
\frac{X_t(x,y)-B(x,y)}
{\max(S(x,y),\epsilon_R)}.
\]

Retain both:

\[
R_t
\qquad\text{and}\qquad
R_t^+ = \max(R_t,0).
\]

Whitening acts on signed \(R_t\). Candidate evidence may apply the frozen positive transformation afterward.

### 11.2 Shrunk covariance

For patch vectors \(p_i\in\mathbb R^d\), estimate:

\[
\widehat C
=
\frac{1}{N-1}
\sum_i
(p_i-\widehat\mu)(p_i-\widehat\mu)^\top.
\]

Use trace shrinkage:

\[
C_\lambda
=
(1-\lambda)\widehat C
+
\lambda
\frac{\operatorname{tr}(\widehat C)}{d}I.
\]

Apply an eigenvalue floor:

\[
\widetilde \nu_j
=
\max
\left(
\nu_j,
\epsilon_{\mathrm{eig}}
\frac{\operatorname{tr}(C_\lambda)}{d}
\right).
\]

### 11.3 Fractional whitening

Define:

\[
W_\gamma
=
U
\operatorname{diag}
\left(
\widetilde\nu_j^{-\gamma/2}
\right)
U^\top,
\qquad
0\leq\gamma\leq1.
\]

Interpretation:

- \(\gamma=0\): centering only;
- \(0<\gamma<1\): partial whitening;
- \(\gamma=1\): full ZCA whitening.

This parameter is central. It lets the data learn how aggressively covariance should be removed rather than forcing a binary choice.

### 11.4 Raw-preserving blend

For the whitened center-coordinate response \(Z\), define:

\[
Y
=
(1-\beta)R
+
\beta Z,
\qquad
0\leq\beta\leq1.
\]

Raw Direct is embedded exactly at \(\beta=0\).

Unit tests must verify bitwise or tolerance-level reproduction of Raw Direct when the operator weight is zero.

### 11.5 Spatial whitening

For each frame:

1. extract centered odd-width spatial patches;
2. fit covariance from deterministic quiet-frame patch samples;
3. apply ZCA/fractional whitening;
4. retain the center coordinate as the scalar output at each pixel;
5. use reflect padding;
6. keep scoring stride one.

The covariance-sampling stride may be greater than one for efficiency, but scoring stride is fixed and is not a scientific hyperparameter.

### 11.6 Temporal whitening

For each pixel:

1. extract centered odd-width temporal windows;
2. fit covariance from deterministic quiet-window samples across pixels;
3. apply fractional whitening;
4. retain the center-time coordinate;
5. use reflect padding for the offline benchmark.

This program is offline. Do not silently reinterpret centered temporal whitening as causal.

### 11.7 Separable spatiotemporal whitening

The default spatiotemporal operator is separable:

\[
W_{st}
=
W_t \otimes W_s.
\]

Apply the spatial and temporal linear transforms without inserting a nonlinearity between them. This avoids estimating a very large joint covariance and provides a controlled test of combined spatial and temporal decorrelation.

### 11.8 Full joint spatiotemporal whitening

Full joint covariance is diagnostic-only.

Permit it only when all conditions hold:

- vector dimension \(d\leq128\);
- nominal quiet sample count at least \(20d\);
- regularized condition number at most \(10^6\);
- effective-rank fraction at least `0.25`;
- eigenspectra are stable across four contiguous quiet blocks;
- smoke memory estimate passes;
- separable whitening has already shown some promise.

If any condition fails, use separable whitening and record:

```text
joint_spatiotemporal_status = rejected_by_conditioning_gate
```

Do not “fix” a failed full covariance by silently increasing shrinkage until every operator becomes nearly identity.

### 11.9 Quiet standardization before mixing

Each operator output must be calibrated to a common quiet scale before mixture:

\[
\widetilde Y_k
=
\frac{
Y_k-\operatorname{median}_{q}(Y_k)
}{
1.4826\operatorname{MAD}_{q}(Y_k)+\epsilon
}.
\]

This is required. Without it, learned mixture weights confound operator utility with arbitrary output magnitude.

---

## 12. QMC and sensitivity-design contract

### 12.1 Terminology

Interpret the requested “Sobolev sampling” as **Sobol sequence sampling**.

The primary design uses a scrambled Sobol low-discrepancy sequence. Latin hypercube sampling is used later as an independent confirmation design.

Do not call the initial procedure “Sobol sensitivity analysis.” A Sobol sequence is a space-filling design. Variance-based Sobol sensitivity indices require a separate pick-freeze/Saltelli-style design and are outside the default program.

### 12.2 Master design

For each continuous operator family:

1. generate one scrambled Sobol sequence of 64 points;
2. use a fixed master sampler seed;
3. write all 64 points to `designs/<family>_sobol_master.tsv` before any scientific evaluation;
4. reveal only nested prefixes of size 8, 16, 32, and, with explicit authorization, 64;
5. never regenerate or reorder the master design after results are observed.

Use `random_base2` semantics and powers of two.

Suggested master seed:

```text
20260819
```

Model seeds remain separate:

```text
screen seeds:       7, 13, 19
confirmation seeds: 7, 13, 19, 29, 37
```

### 12.3 Parameter transforms

Use:

- linear scaling for bounded strengths and blend coefficients;
- log scaling for learning rates, eigenvalue floors, and small regularizers;
- deterministic mapping to odd support widths;
- separate designs for categorical operator families.

Do not encode operator family as an arbitrary continuous number.

### 12.4 Initial operator domains

#### Spatial family

| Parameter | Domain | Mapping |
|---|---:|---|
| spatial width | 3–21 odd pixels | quantized from Sobol coordinate |
| whitening exponent \(\gamma_s\) | 0–1 | linear |
| shrinkage \(\lambda_s\) | \(10^{-4}\)–0.5 | logit/log-like monotone mapping |
| eigen floor ratio | \(10^{-6}\)–\(10^{-2}\) | log |
| whitening blend \(\beta_s\) | 0.10–1.0 | linear |
| explicit controls | \(\beta=0,\gamma=0,\gamma=1\) | appended, not sampled |

#### Temporal family

| Parameter | Domain | Mapping |
|---|---:|---|
| temporal width | 3–33 odd frames | quantized |
| whitening exponent \(\gamma_t\) | 0–1 | linear |
| shrinkage \(\lambda_t\) | \(10^{-4}\)–0.5 | log-like |
| eigen floor ratio | \(10^{-6}\)–\(10^{-2}\) | log |
| whitening blend \(\beta_t\) | 0.10–1.0 | linear |
| explicit controls | \(\beta=0,\gamma=0,\gamma=1\) | appended |

#### Separable spatiotemporal family

| Parameter | Domain |
|---|---:|
| spatial width | 3–15 odd pixels |
| temporal width | 3–25 odd frames |
| spatial whitening exponent | 0–1 |
| temporal whitening exponent | 0–1 |
| shared or separate shrinkage | start shared; separate only in rescue |
| raw-preserving blend | 0.10–1.0 |
| eigen floor ratio | \(10^{-6}\)–\(10^{-2}\) |

Keep the initial dimensionality small. Do not include pooling temperature, NMS radius, match radius, candidate budget, or arbitrary nonlinearities in this design.

### 12.5 Quantization and duplicate handling

Mapping Sobol coordinates to odd widths can create duplicate configurations.

Codex must:

- generate the full continuous design first;
- apply deterministic quantization;
- deduplicate exact operator specifications;
- preserve the earliest design index;
- report duplicate count;
- continue revealing later Sobol points until the requested number of unique configurations is reached;
- never replace duplicates with random points.

### 12.6 Label-free anchor-bank construction

Do not select mixture anchors by label performance.

From numerically valid sampled operators, compute a small signature using:

- normalized operator parameters;
- pairwise output correlation on a fixed label-free probe tensor;
- quiet variance ratio;
- effective-rank fraction;
- log condition number.

Select at most 12 total anchors, including Raw Direct, using deterministic farthest-point or medoid selection in this signature space.

Recommended maximum:

```text
Raw Direct:                 1
spatial anchors:            3–4
temporal anchors:           3–4
spatiotemporal anchors:     3–4
total:                      <= 12
```

### 12.7 Latin-hypercube confirmation

Use one independent scrambled LHS design only when:

- a family survives the Sobol screen;
- the inferred response surface is being used to justify scale sensitivity;
- or the winning region lies near a Sobol-domain boundary.

Default confirmation size:

```text
16 unique LHS points
```

Use a distinct fixed seed. Do not merge LHS cells into the training candidate bank after seeing results. LHS is a robustness check.

### 12.8 Sensitivity summaries

After at least 16 valid points, report:

- parameter versus primary-metric Spearman correlations;
- top-quartile parameter ranges;
- whether the best point is at a domain boundary;
- a cross-validated low-capacity surrogate \(R^2\), if stable;
- output-correlation clusters;
- condition-number relationship with performance.

After at least 32 valid points, optional model-agnostic permutation importance may be reported.

Do not report variance-based Sobol indices unless a separate valid design is implemented.

---

# 13. Stage S0 — evaluator freeze and baseline audit

## 13.1 Goal

Establish an immutable evaluator and reproduce the current known baselines before implementing learned selection.

## 13.2 Required lanes

Run or reconstruct:

1. Raw Direct;
2. current best fixed whitening lane, if one is already explicitly identified;
3. current ICA lane under its existing frozen settings;
4. current whitening→ICA lane, if it exists as a reproducible lane;
5. archived amplitude PCA rank 8 metrics as a context comparator, without treating it as part of the whitening bank.

If the “current best whitening” is not uniquely specified in the repository, record that fact. Do not select it post hoc using all four bursts. Instead, include the documented fixed configuration and later construct a leakage-safe Fixed-Select comparator from the new bank.

## 13.3 Reproduction invariants

Raw Direct must reproduce:

```text
Q1 mean burst recall:       0.6056159420289855
Q1 total matches:           49 / 79
Q1 event candidates:        232
fixed-budget total matches: 52 / 79
fixed-budget mean recall:   approximately 0.6572
```

In addition:

- candidate ranking must be deterministic;
- frame and coordinate projection must match;
- top-58 candidate hashes must remain stable;
- no labels may affect quiet normalization;
- fixed-budget and Q1 metrics must be clearly separated.

## 13.4 Unit and integration work before the full video

Implement:

- synthetic peak-matching fixture;
- exact UI-to-NumPy frame conversion test;
- deterministic tie-order test;
- one-to-one matching test;
- temporary tiny-video evaluation;
- output-collision test;
- preflight resource estimate.

## 13.5 S0 decision

### Advance

Advance when:

- all invariants reproduce;
- coordinate overlay passes;
- candidate ranking is deterministic;
- output roots are collision-safe;
- metric schema validates.

### Diagnostic rescue

Use one rescue when only a deterministic implementation discrepancy exists, such as:

- score tie ordering;
- frame conversion;
- label parsing;
- quiet pseudo-burst construction;
- temporal pooling mismatch.

### Stop program

Stop when:

- the raw baseline cannot be reproduced after one bounded diagnostic;
- labels or video fingerprints do not match the documented dataset;
- evaluation leakage is detected;
- coordinate projection is invalid.

No later scientific result is valid until S0 passes.

---

# 14. Stage S1 — operator numerical and response screen

## 14.1 Goal

Determine whether any well-conditioned spatial, temporal, or separable spatiotemporal whitening regime has useful signal, and determine whether operators provide complementary behavior.

This stage evaluates fixed operators. It does not train a gating network.

## 14.2 S1A synthetic numerical validation

Create synthetic fixtures with:

- correlated Gaussian spatial patches;
- correlated temporal AR processes;
- separable spatiotemporal covariance;
- sparse positive transients;
- one ill-conditioned covariance;
- one rank-deficient covariance.

Verify:

- \(\gamma=0\) reduces to centering;
- \(\gamma=1\) approximately whitens valid synthetic covariance;
- shrinkage lowers the condition number;
- fractional whitening interpolates continuously;
- separable whitening agrees with explicit Kronecker whitening on a tiny tensor;
- Raw Direct is exact at \(\beta=0\);
- gradients are finite where differentiability is required later;
- ill-conditioned cases stop or shrink safely.

No biological run proceeds until these pass.

## 14.3 S1B full-video sanity prefix

Run the first 8 unique Sobol configurations per family.

This is deterministic fixed-operator evaluation, so model seeds are not required.

Store only essential metrics and operator parameters.

### Early family stop at 8 points

Stop a family immediately only when all are true:

- all valid operators are at least `0.10` below Raw Direct in Macro-KPR@58;
- no operator improves any burst by at least one known match;
- operator outputs are strongly redundant with one another;
- no promising boundary trend is present;
- no conditioning issue plausibly explains the result.

Otherwise continue to 16 points.

## 14.4 S1C primary response screen

Reveal 16 unique Sobol configurations for surviving families.

Compute:

1. leakage-safe Fixed-Select performance;
2. post-hoc fixed oracle;
3. per-burst best operators;
4. pairwise output correlations;
5. sensitivity summaries;
6. numerical-health summaries.

### Family classifications

#### Promising

A family is promising when Fixed-Select versus Raw Direct has:

- `Δ Macro-KPR@58 >= +0.02`;
- at least `+2` pooled known matches;
- nonnegative delta in at least three of four bursts;
- numerical-health pass.

#### Complementary

A family is complementary when it does not meet the global promising gate, but:

- the per-burst oracle exceeds Raw Direct by at least `0.04` macro recall or four pooled matches; and
- different operators win in different bursts; and
- output diversity is genuine rather than scale duplication.

A complementary family may enter the mixture bank even if no single operator promotes.

#### Weak but informative

A family is weak but informative when:

- best leakage-safe delta lies between `-0.02` and `+0.02`; or
- gains are confined to one or two bursts; or
- performance is strongly related to conditioning or scale.

Retain at most two diagnostic anchors. Do not widen automatically.

#### Dead

Stop the family when:

- Fixed-Select is worse than Raw Direct by at least `0.02`;
- the oracle offers no material gap;
- no burst-specific complementarity exists;
- and numerical health is acceptable.

## 14.5 S1D optional 32-point extension

Expand a family from 16 to 32 points only if at least one is true:

- the best region lies at a search boundary;
- the family is complementary but under-resolved;
- the low-capacity surrogate has poor fit and high local uncertainty;
- conditioning changes sharply over the domain;
- Fixed-Select has a positive but sub-gate delta;
- two separated promising regions exist.

Do not reveal 64 points without explicit user authorization.

## 14.6 S1 diagnostic rescue

One rescue is allowed per family.

### Conditioning rescue

Use when performance failure correlates with:

- condition number;
- rank deficiency;
- unstable quiet-block spectra;
- non-finite values.

Permitted changes:

- increase minimum shrinkage;
- reduce maximum support;
- increase eigenvalue floor within declared bounds;
- replace full joint covariance with separable covariance;
- use deterministic block-subsampled covariance.

Rerun at most 8 diagnostic points.

### Calibration rescue

Use when individual operators appear useful but mixtures or rankings are dominated by output scale.

Permitted changes:

- fix quiet median/MAD standardization;
- verify sign and center-coordinate readout;
- verify candidate ranking and pooling.

Do not change the biological metric.

### No rescue

Do not rescue a well-conditioned family that simply fails to improve known-positive recovery.

## 14.7 S1 decision

Advance to S2 when at least one family is promising or complementary.

If all families are dead, skip learned mixture development and proceed only to the minimal ICA confirmation in S5.

---

# 15. Stage S2 — global learned operator mixture

## 15.1 Scientific question

Can a learned low-dimensional combination of statistically motivated operators outperform selecting one fixed operator from the same label-free bank?

## 15.2 Model

Let standardized operator outputs be \(\widetilde Y_k\), including Raw Direct as \(k=0\).

Learn global logits \(a_k\):

\[
\alpha_k
=
\frac{e^{a_k/\tau_\alpha}}
{\sum_j e^{a_j/\tau_\alpha}},
\]

\[
Z
=
\sum_{k=0}^{K-1}
\alpha_k \widetilde Y_k.
\]

Constraints:

- \(K\leq12\);
- no spatially varying weights;
- no absolute coordinates;
- no hidden CNN;
- no learned NMS, threshold, or pooling;
- Raw Direct must be one eligible anchor;
- operator outputs are frozen in S2.

## 15.3 Supervised objective

Reuse the existing weakly supervised multiple-instance ranking structure.

For labeled occurrence \(i\) in burst \(b\):

\[
s_{b,i}^{+}
=
\operatorname{LME}
\left\{
Z_t(p):
t\in W_b,\;
p\in D_4(p_i)
\right\}.
\]

Construct quiet bags with matched durations, preferably at the same coordinates plus deterministic additional quiet locations.

Use:

\[
\mathcal L_{\mathrm{rank}}
=
\operatorname{softplus}
\left(
m-s^{+}+s^{q}
\right).
\]

Add only small, bounded regularizers:

\[
\mathcal L
=
\mathcal L_{\mathrm{rank}}
+
\lambda_H R_H(\alpha)
+
\lambda_R R_{\mathrm{raw}}(\alpha).
\]

Possible regularizers:

- entropy or negative entropy to test diffuse versus sparse selection;
- weak raw-anchor prior to prevent immediate destructive departure.

Do not add both many regularizers and a large search space.

## 15.4 Training-hyperparameter Sobol design

Generate a separate 8-point nested Sobol pilot over:

| Parameter | Domain |
|---|---:|
| learning rate | \(10^{-4}\)–\(10^{-1}\), log |
| alpha temperature | 0.25–2.0 |
| entropy regularization | \(10^{-5}\)–\(10^{-1}\), log, signed mode fixed per variant |
| raw-anchor regularization | \(10^{-5}\)–\(10^{-1}\), log |
| ranking margin | 0.1–2.0 |

Keep optimizer family fixed. Use Adam for the pilot unless the existing learnable-contrast optimizer contract requires another tested default.

Do not search optimizer family, batch size, NMS, pooling, and operator bank simultaneously.

## 15.5 Initialization controls

Evaluate two initializations in the sanity stage:

1. Raw-dominant:
   ```text
   alpha_raw ≈ 0.8
   remaining mass distributed uniformly
   ```
2. Uniform.

If they converge to the same solution, retain Raw-dominant for later stages. If they disagree materially, treat optimization as unstable and diagnose before promotion.

## 15.6 S2A smoke

Run:

- one outer fold;
- one seed;
- Raw-dominant and uniform initialization;
- two pilot hyperparameter cells;
- reduced epochs.

Verify:

- held-out labels are inaccessible to the trainer;
- gradients affect logits;
- weights sum to one;
- Raw anchor can remain dominant;
- validation metrics are computed correctly;
- only minimal artifacts are written.

This smoke has no scientific interpretation.

## 15.7 S2B pilot

For each outer fold:

- evaluate the 8 Sobol training cells through inner validation;
- use one model seed;
- select one cell inside the outer fold;
- refit on all three outer-training bursts;
- evaluate the held-out burst.

Pilot advancement is deliberately relaxed.

### Advance to seed confirmation when

- median delta versus Fixed-Select is positive;
- at least two bursts are nonnegative;
- no burst loses more than `0.15` recall;
- numerical health passes;
- the learned solution is not purely an output-scale artifact.

### Diagnostic rescue when

- training ranking loss improves but inner and outer metrics do not;
- initialization changes the result materially;
- weights collapse onto a numerically pathological operator;
- operator calibration is inconsistent;
- the fixed oracle suggests strong complementarity that the mixture fails to capture.

### Stop when

- delta is `<= -0.02`;
- no per-burst complementarity is recovered;
- Fixed-Select is already no better than Raw Direct;
- and no numerical or calibration cause is present.

## 15.8 S2C three-seed confirmation

Run only the top inner-selected training design with seeds:

```text
7, 13, 19
```

across all four outer folds.

Advance to final confirmation when:

- median-seed Macro-KPR@58 is above Fixed-Select;
- at least two of three seeds are nonnegative overall;
- at least three bursts are nonnegative after seed aggregation;
- learned weight patterns are not arbitrary across seeds.

Weight stability should be summarized with:

- median weight vector;
- per-anchor interquartile range;
- Jensen–Shannon divergence across seed solutions;
- effective number of active operators:
  \[
  K_{\mathrm{eff}} = \exp(H(\alpha)).
  \]

Only the weight vector and compact stability summary are retained, not full trajectories.

## 15.9 S2D five-seed final confirmation

Run:

```text
7, 13, 19, 29, 37
```

for the single selected design.

### Strong promotion gate

Promote the global mixture when all hold:

- `Δ Macro-KPR@58 >= +0.02` versus Fixed-Select;
- at least `+2` pooled known matches;
- nonnegative burst delta in at least three of four bursts;
- at least four of five seed-level aggregate deltas are nonnegative;
- numerical health passes;
- no catastrophic trace-preservation concern;
- hidden-identity guardrail does not show obvious memorization.

### Research-interest only

Record but do not replace the baseline when:

- delta is between `+0.01` and `+0.02`; or
- only one additional known occurrence is recovered; or
- gain is limited to two bursts; or
- weight stability is poor.

This result may justify analysis but does not earn a more complex adaptive model by itself.

### Stop

Stop learned mixtures when:

- final median delta is nonpositive;
- the confidence pattern is seed-dependent;
- Fixed-Select is equally good;
- or the gain disappears under hidden-identity evaluation.

## 15.10 S2 optimization rescue

One rescue is allowed.

Use a rescue only after identifying one of:

- validation objective is mis-scaled;
- raw anchor is improperly standardized;
- softmax temperature causes immediate saturation;
- learning rate is clearly unstable;
- operator outputs are collinear and need bank compression.

Allowed rescue:

- compress redundant anchors;
- narrow learning-rate range;
- fix standardization;
- strengthen a weak raw prior;
- use deterministic projected/simplex optimization for the global weights.

Do not add a neural gating network as an “optimization fix.”

---

# 16. Stage S3 — continuous learned scale

## 16.1 Entry conditions

Enter S3 only if one of the following is true:

1. S2 strongly promotes; or
2. S1 shows a smooth scale-response relationship and a compact optimum; or
3. S2 matches the best fixed bank while using several adjacent scales, suggesting an interpolable optimum.

Do not enter S3 merely because continuous scale is theoretically attractive.

## 16.2 Scientific question

Can a small continuous parameterization replace discrete scale selection while preserving or improving biological utility?

## 16.3 Initial model

Begin with the single winning operator family.

Learn only:

- one spatial scale, or one temporal scale, or one spatial and one temporal scale;
- whitening exponent;
- raw-preserving blend.

Keep shrinkage and eigen floor fixed at the best well-conditioned values from S1.

Do not initially learn:

- local scale maps;
- separate covariance per ROI;
- multiple hierarchy levels;
- a large kernel;
- pooling or candidate parameters.

## 16.4 Differentiable implementation order

1. Implement differentiable synthetic covariance and inverse-square-root tests.
2. Implement one scalar learned scale.
3. Confirm finite gradients through eigendecomposition.
4. Add whitening exponent.
5. Add raw blend.
6. Add learnable shrinkage only if conditioning analysis shows it is necessary.

For discrete patch extraction, choose one of two explicit implementations:

### Preferred direct path

Use differentiable interpolation or continuous kernels to define weighted covariance support.

### Acceptable bounded surrogate

Interpolate between adjacent precomputed well-conditioned anchor operators with a continuous scale coordinate.

If using the surrogate, label it accurately as continuous interpolation over an operator bank, not direct continuous covariance learning.

## 16.5 Boundary behavior

If the learned scale reaches a domain boundary in more than 75% of outer-fold/seed fits:

- inspect gradient direction and conditioning;
- expand that boundary once by at most 50%;
- rerun only the finalist;
- stop if the optimum remains at the expanded boundary or numerical health worsens.

Do not repeatedly widen the domain.

## 16.6 S3 pilot

Run:

- all four outer folds;
- three seeds;
- one family;
- at most two initialization scales.

Advance if:

- the continuous model remains within `0.01` Macro-KPR@58 of the promoted S2 mixture;
- it remains at least `0.02` above Fixed-Select;
- scale estimates are finite;
- scale coefficient of variation across seeds/folds is at most `0.25`, unless true burst heterogeneity is demonstrated;
- solutions are not pinned to bounds.

Noninferiority is an acceptable success because the continuous model is simpler and avoids maintaining a discrete bank.

## 16.7 S3 final decision

### Promote continuous scale

When it is noninferior to the mixture, superior to Fixed-Select, stable, and computationally acceptable.

### Retain mixture

When continuous scale underperforms S2 by more than `0.01` but S2 remains useful.

### Stop continuous learning

When scale is unidentifiable, boundary-pinned, poorly conditioned, or unstable across seeds.

### Evidence for adaptivity

Flag S4 as scientifically justified only when:

- learned global scale varies systematically by held-out burst;
- per-burst fixed-scale winners are separated;
- or the per-burst oracle materially exceeds the best global scale.

---

# 17. Stage S4 — adaptive operator selection

## 17.1 Strict entry gate

S4 is not a default stage.

Enter only if at least one holds:

- per-burst oracle exceeds the global model by at least `0.04` Macro-KPR@58;
- at least four pooled known matches are available to a context-specific oracle;
- different scale regions win in at least three bursts;
- global learned weights vary systematically across folds;
- labeled and label-free diagnostics both indicate spatial or temporal heterogeneity.

Do not build a local gating network when the only evidence is that a global model is slightly imperfect.

## 17.2 Capacity ladder

Proceed from smaller to larger adaptivity.

### S4A segment-adaptive mixture

Predict one weight vector per temporal segment from global pooled Raw Direct context.

Constraints:

- no burst ID input;
- no absolute frame index;
- fewer than 1,000 trainable parameters;
- weights constant within the segment.

### S4B block-adaptive mixture

Predict weights on coarse non-overlapping or overlapping spatiotemporal blocks, then interpolate smoothly.

### S4C dense local mixture

Only if S4B promotes.

Suggested maximum gating network:

```text
input: Raw Direct residual only
Conv3d: 1 -> 8, kernel (3,5,5)
activation: GELU
Conv3d: 8 -> K, kernel 1
softmax over K operators
```

Keep total trainable parameter count below 10,000.

Do not feed labels, coordinates, ROI masks, or candidate maps to the gate.

## 17.3 Regularization

Use:

\[
\mathcal L
=
\mathcal L_{\mathrm{rank}}
+
\lambda_{tv}
\left(
\|\nabla_{xy}\alpha\|_1
+
\eta_t\|\nabla_t\alpha\|_1
\right)
+
\lambda_H R_H(\alpha)
+
\lambda_R R_{\mathrm{raw}}(\alpha).
\]

The gate should not flicker arbitrarily or select different operators pixel-by-pixel without evidence.

## 17.4 Sequential run schedule

### Sanity

- one outer fold;
- one seed;
- segment-adaptive only.

### Pilot

- all four outer folds;
- one seed;
- segment-adaptive and block-adaptive.

### Confirmation

- only the winning capacity;
- three seeds;
- hidden-identity evaluation mandatory.

### Final

- five seeds only if confirmation passes.

## 17.5 Promotion gate

Adaptive selection must beat the promoted global model, not merely Raw Direct.

Require:

- `Δ Macro-KPR@58 >= +0.02` versus the global model;
- `Δ Macro-KPR@58 >= +0.03` versus Fixed-Select;
- at least `+2` pooled matches versus global;
- nonnegative delta in at least three bursts;
- hidden-identity result does not collapse;
- weight maps are smooth and reproducible;
- effective operator use is not an arbitrary high-entropy mixture everywhere;
- full finalist audit passes.

If the adaptive method ties the global method, retain the global method.

## 17.6 Stop conditions

Stop adaptivity when:

- training improves but held-out bursts do not;
- hidden identities fail;
- gates encode location-specific memorization;
- weight maps are unstable across seeds;
- gain depends on one burst;
- the global model is within `0.01`;
- conditioning problems recur locally.

Do not respond to overfitting by adding more data augmentations and architecture variants indefinitely.

---

# 18. Stage S5 — ICA marginal-utility test

## 18.1 Scientific question

After controlling for the representation and whitening, does an independence-seeking rotation recover biologically useful evidence that matched non-ICA rotations do not?

The relevant quantity is:

\[
\Delta_{\mathrm{ICA}}
=
\operatorname{MacroKPR@58}
(\text{representation + ICA})
-
\operatorname{MacroKPR@58}
(\text{same representation, no ICA}).
\]

## 18.2 Entry behavior

Run a minimal S5 even if learned whitening fails, because the overall project still needs a clear ICA decision. However:

- if S1 and S2 fail, use only the existing Raw/PCA-whitened representation and a small rank-8/16 confirmation;
- do not launch a broad new ICA search;
- do not test rank 64.

If learned whitening promotes, use its frozen output as the ICA input.

## 18.3 ICA tracks

### Track A: full-window spatial ICA

Treat pixel traces over the 560-frame review interval as observations/features according to the existing representation benchmark.

Use ranks:

```text
8 and 16
```

If rank 8 is unsupported by a specific implementation, use rank 16 only and document why.

### Track B: pairwise temporal ICA

Use only as a separate diagnostic when the model is explicitly two-frame or adjacent-frame.

A component aligned with \([-1,1]\) is a temporal-derivative-like feature. Do not claim source separation merely because the objective is ICA.

Do not combine Track A and Track B conclusions.

## 18.4 Matched controls

For each input and rank, evaluate:

1. no rotation / identity in the whitened subspace;
2. PCA component evidence;
3. three deterministic random orthogonal rotations;
4. FastICA;
5. CS-Parzen ICA, if numerically authorized;
6. InfoMax/Bell ICA only if a maintained implementation exists with a matched evidence contract.

All methods must use:

- the same rank;
- the same sample set;
- the same whitening input;
- the same component orientation rule;
- the same component-evidence construction;
- the same temporal pooling;
- the same candidate budget;
- the same matching.

## 18.5 ICA essential diagnostics

Retain:

- convergence flag and iteration count;
- aligned component stability across seeds;
- objective value;
- whitened covariance condition number;
- derivative-angle distance for pairwise ICA;
- primary biological metric;
- trace-preservation metric;
- runtime and peak memory.

Do not retain every component image for screening.

## 18.6 Sequential ICA schedule

### S5A synthetic and tiny smoke

Verify:

- permutation/sign alignment;
- random-rotation controls;
- stable deterministic evidence construction;
- ICA can recover known synthetic independent components;
- derivative alignment is correctly identified in the two-frame case.

### S5B pilot

Use:

```text
ranks: 8, 16
seeds: 7, 13, 19
objectives: no-rotation, random rotation, FastICA, CS-Parzen
```

If CS-Parzen is substantially more expensive, run it only at the rank that passed FastICA/no-rotation screening.

### S5C finalist confirmation

Use one rank and one objective with five seeds.

## 18.7 ICA promotion gate

ICA is scientifically useful only when all hold:

- `Δ Macro-KPR@58 >= +0.02` versus matched no-ICA;
- at least `+2` pooled known matches;
- nonnegative delta in at least three bursts;
- it beats the random-rotation distribution, not merely one random seed;
- at least 90% of fitted cells converge;
- aligned component stability is high enough to interpret;
- gain survives hidden-identity evaluation when the model is supervised downstream;
- gain is not solely at the quiet-threshold operating point.

A recommended stability target for rank 8/16 is mean aligned absolute component correlation at least `0.90`, interpreted alongside component ambiguity.

## 18.8 ICA stop conclusions

### Stop: independence not useful

Use when ICA improves its independence objective but does not improve Macro-KPR@58.

Conclusion:

> The independence criterion changes the representation but does not provide incremental known-event recovery under a matched detection contract.

### Stop: generic rotation sufficient

Use when random orthogonal rotations match ICA.

Conclusion:

> The gain is attributable to basis rotation or component aggregation, not specifically to independence.

### Reclassify: derivative feature

Use when adjacent-frame ICA consistently aligns with the analytic difference direction.

Conclusion:

> In this regime, ICA behaves primarily as a learned temporal-derivative operator rather than a multi-source separator.

### Stop: unstable ICA

Use when convergence, rank, sign/permutation alignment, or seed stability fails.

Do not increase rank or objective complexity as the first response.

---

# 19. Stage-level promotion and stop matrix

| Observation | Interpretation | Action |
|---|---|---|
| No fixed operator approaches Raw Direct; oracle gap small | Whitening family lacks task utility | Stop family |
| Fixed oracle is strong; leakage-safe Fixed-Select weak | Heterogeneity or selection overfit | Consider mixture; do not claim fixed improvement |
| Fixed-Select strong; mixture ties | Learning does not add value | Retain fixed operator |
| Uniform mix equals learned mix | Ensembling, not learning, explains gain | Retain simpler uniform mix or fixed lane |
| Learned mix beats fixed and is stable | Global learned selection is useful | Advance to S3 or S5 |
| Learned mix beats fixed but weights unstable | Non-identifiability or redundant bank | Compress bank; one rescue |
| Continuous scale matches mixture | Discrete bank unnecessary | Promote continuous scale |
| Continuous scale is boundary-pinned | Domain or identifiability issue | Expand once, then stop |
| Per-burst oracle greatly exceeds global | Real heterogeneity is plausible | Permit S4 |
| Adaptive model ties global | Complexity not earned | Retain global |
| Adaptive model fails hidden identities | Memorization/overfit | Stop adaptive |
| ICA objective improves but detection does not | Independence is not the useful assumption | Stop ICA |
| Random rotation matches ICA | Rotation/aggregation effect | Stop ICA-specific claim |
| Pairwise ICA aligns with difference | Temporal derivative | Reclassify |
| Failures correlate with condition number | Numerical failure | One conditioning rescue |
| Well-conditioned method remains worse | Scientific failure | Stop; no rescue |

---

# 20. Early-stopping rules

## 20.1 Fit-level numerical stop

Stop immediately on:

- NaN or Inf loss;
- non-finite operator output;
- non-finite parameters;
- eigendecomposition failure;
- condition number above hard safety bound after permitted regularization;
- resource-limit violation;
- output collision;
- data fingerprint mismatch.

Write `failure.json` and preserve the small operational log.

## 20.2 Learned-model early stopping

Default:

```text
max epochs:             200
validation interval:    every 5 epochs
patience:               6 validation checks
minimum metric delta:   one tie-breaking unit in validation score
```

Select best epoch lexicographically by:

1. higher validation KPR@58;
2. lower validation ranking loss;
3. lower complexity penalty;
4. earlier epoch.

Because KPR@58 is discrete, ranking loss acts only as a tie breaker.

### Futility stop

After at least 30% of the epoch budget, stop when:

- training objective improves;
- validation KPR remains at least `0.10` below Raw Direct;
- no prior validation checkpoint was competitive;
- numerical health is normal.

### Saturation stop

Stop when the weight vector changes less than a small declared tolerance for the entire patience window and validation does not improve.

Do not retain full per-epoch history. Retain:

- best epoch;
- last epoch;
- best validation metric;
- best validation ranking loss;
- stop reason;
- final/best compact parameter vector.

## 20.3 Stage-level futility

At a pilot stage, stop expansion when:

- median delta is negative;
- the best plausible diagnostic comparator is also weak;
- no numerical issue exists;
- and the next stage would only increase capacity.

A stage-level stop must be recorded even when all jobs completed successfully.

---

# 21. Essential metrics and artifact contract

## 21.1 One row per fit

`runs.tsv` should contain only:

```text
run_id
stage
method
operator_family
design_id
outer_fold
inner_fold
seed
status
stop_reason
heldout_burst_id
heldout_labels
heldout_matches_at_58
heldout_recall_at_58
heldout_matches_q1
heldout_candidates_q1
trace_preserve
max_condition_number
effective_rank_fraction
converged
best_epoch
runtime_seconds
peak_rss_mib
peak_vram_mib
parameter_file
config_hash
git_commit
data_fingerprint
```

Do not place large arrays in TSV cells.

## 21.2 Compact aggregate summary

`stage_summary.json` contains:

- primary aggregate metric;
- comparator metrics;
- pooled matches;
- per-burst vector;
- seed distribution;
- Q1 diagnostic;
- trace preservation;
- numerical health;
- selected design IDs;
- learned weight/scale summary;
- decision link.

## 21.3 Parameter storage

For global mixture and continuous scale, store:

```text
parameters/<run_id>.json
```

These files should be tiny and include only:

- logits or normalized weights;
- scales;
- whitening exponents;
- shrinkage;
- raw blend;
- regularization values;
- seed;
- best epoch.

No full model checkpoint is needed for a non-promoted global mixture if the model can be reconstructed from this file.

For adaptive networks:

- use an atomic temporary resumable checkpoint during training;
- retain `best_state.pt` only for promoted or still-active confirmation runs;
- non-promoted runs retain a hash and compact architecture/metric metadata, not every epoch checkpoint.

## 21.4 Operational files

Permit:

```text
progress.jsonl
failure.json
resource_summary.json
```

`progress.jsonl` should contain stage transitions and bounded heartbeats, not per-batch logs.

## 21.5 Candidate artifacts

For screening runs:

- store candidate ranking hash;
- store top-candidate tables only for selected comparators and finalists.

For finalists:

- store complete top-58 candidates per burst;
- Q1 candidates;
- known matches;
- model-only predictions;
- full scientific audit.

## 21.6 Prohibited default outputs

Do not generate for every run:

- full score videos;
- TIFF stacks;
- every component map;
- every kernel image;
- per-epoch loss plots;
- full covariance matrices;
- dense operator outputs;
- exhaustive candidate CSVs;
- all intermediate checkpoints.

Generate these only for a promoted finalist or a bounded diagnostic rescue that explicitly needs them.

---

# 22. Preflight contract

Preflight must calculate and write:

- source and label fingerprints;
- frame and coordinate overlay;
- exact stage requested;
- total design points and unique points;
- expected fit count;
- expected operator-cache size;
- peak RAM estimate;
- peak VRAM estimate;
- output-size estimate;
- disk headroom;
- GPU availability;
- active-process summary;
- collision checks;
- scientific-audit resolution;
- prerequisite stage decision;
- whether diagnostic rescue remains available;
- exact command required to run.

Preflight must be read-only with respect to completed outputs.

Before a long run, inspect live system state rather than trusting stale documentation.

---

# 23. Testing plan

## 23.1 Operator unit tests

- identity/centering at \(\gamma=0\);
- approximate covariance identity at \(\gamma=1\);
- monotone condition-number improvement with shrinkage on a fixture;
- eigen-floor behavior;
- fractional-whitening continuity;
- raw-blend exactness at \(\beta=0\);
- spatial shape and padding;
- temporal alignment;
- separable versus explicit Kronecker on a tiny fixture;
- full-joint rejection gate;
- deterministic covariance sampling;
- finite gradients.

## 23.2 Design unit tests

- scrambled Sobol reproducibility;
- nested 8/16/32 prefixes;
- no regeneration after results;
- odd-width quantization;
- duplicate replacement by later master points;
- separate categorical-family designs;
- LHS independence;
- parameter transform bounds;
- design hash stability.

## 23.3 Evaluation unit tests

- Raw Direct exact metric contract;
- deterministic score ties;
- six-pixel NMS;
- six-pixel one-to-one matching;
- fixed 58 candidate truncation;
- UI/NumPy frame conversion;
- per-burst macro versus pooled metric;
- unknown candidate terminology;
- hidden-identity split reproducibility;
- held-out leakage assertion.

## 23.4 Training unit tests

- alpha simplex;
- raw-dominant initialization;
- uniform initialization;
- quiet standardization;
- ranking-bag construction;
- held-out labels inaccessible;
- best-epoch selection;
- patience and futility stops;
- tiny parameter serialization;
- exact reconstruction from parameter JSON.

## 23.5 Decision unit tests

Test every boundary:

- `+0.019` does not satisfy a `+0.02` gate;
- `+2` matches but only two winning bursts does not strongly promote;
- unstable seeds trigger caution or stop;
- conditioning failure triggers rescue only once;
- no oracle gap triggers branch stop;
- adaptive stage refuses without heterogeneity evidence;
- ICA refuses rank 64;
- final audit refuses metrics-only mode.

## 23.6 CLI and integration tests

- preflight on a tiny fixture;
- stage refusal without prerequisite decision;
- output collision refusal;
- resume after interrupted temporary run;
- no overwrite of completed root;
- lazy import and thread bounds;
- minimal output inventory;
- finalist audit re-enabled;
- report and JSON agreement.

## 23.7 Synthetic scientific fixture

Build a small synthetic video with:

- known correlated background;
- sparse transient events;
- one known spatial correlation scale;
- one known temporal correlation scale;
- optional independent sources.

The synthetic test should verify that:

- fixed whitening can reduce known covariance;
- raw-preserving blend avoids catastrophic event removal;
- global scale learning can recover a neighborhood near the true scale;
- ICA only promotes when independent sources actually exist;
- a derivative-like pairwise source is classified as derivative-like.

Synthetic success does not authorize biological promotion. It only validates implementation.

---

# 24. Codex work packages and delegation

Use parallel work only where file ownership does not overlap.

## Work package A — evaluation and invariants

**Owns:**

- canonical evaluator wrapper;
- metric definitions;
- Raw Direct reproduction;
- split and leakage assertions;
- essential metric schema.

**Primary files:**

```text
neurobench/experiments/learned_operator_selection/evaluation.py
neurobench/experiments/learned_operator_selection/data.py
tests/test_learned_operator_evaluation.py
```

**Checkpoint A:**

Raw Direct reproduction passes before other work is accepted.

## Work package B — whitening operators

**Owns:**

- shrinkage;
- eigen flooring;
- fractional ZCA;
- spatial, temporal, and separable spatiotemporal operators;
- numerical diagnostics;
- synthetic tests.

**Primary files:**

```text
neurobench/algorithms/local_whitening.py
neurobench/experiments/learned_operator_selection/operators.py
tests/test_local_whitening.py
```

**Checkpoint B:**

All synthetic covariance and raw-blend tests pass.

## Work package C — QMC design and anchor bank

**Owns:**

- Sobol master design;
- parameter transforms;
- duplicate handling;
- label-free operator signatures;
- anchor selection;
- LHS confirmation.

**Primary files:**

```text
neurobench/experiments/learned_operator_selection/design.py
tests/test_learned_operator_design.py
```

**Checkpoint C:**

Master design hashes and nested prefixes are deterministic.

## Work package D — learned models and early stopping

**Owns:**

- global mixture;
- continuous scale;
- optional adaptive ladder;
- ranking loss;
- compact checkpointing;
- early stopping.

**Primary files:**

```text
neurobench/algorithms/learned_operator_mixture.py
neurobench/experiments/learned_operator_selection/training.py
```

**Checkpoint D:**

Tiny synthetic training recovers a useful operator and serializes minimal parameters.

## Work package E — orchestration, decisions, reports

**Owns:**

- strict config;
- preflight;
- stage state machine;
- decision gates;
- CLI;
- output inventory;
- compact report.

**Primary files:**

```text
neurobench/experiments/learned_operator_selection/config.py
preflight.py
runner.py
decisions.py
report.py
neurobench/cli/experiment.py
```

**Checkpoint E:**

No stage can run out of order or overwrite an output.

## Work package F — ICA ablation

Start only after A–E are integrated.

**Owns:**

- matched no-rotation/random-rotation/ICA controls;
- rank-8/16 limits;
- stability and derivative-alignment diagnostics.

**Primary files:**

```text
neurobench/experiments/learned_operator_selection/ica_ablation.py
tests/test_learned_operator_experiment.py
```

**Checkpoint F:**

Synthetic ICA controls distinguish independence from generic rotation.

## Integration owner

One agent must own final integration and must:

- resolve interfaces;
- run focused tests;
- inspect diffs;
- ensure no unrelated files changed;
- verify documentation and example manifest;
- produce a dry-run command list.

Do not stage, commit, push, or open a PR without separate user authorization.

---

# 25. Implementation milestones

## Milestone 1 — documentation and config skeleton

Deliver:

- this handoff in the repository;
- workflow document;
- strict config classes;
- example manifest;
- stage enums;
- no scientific algorithms yet.

Acceptance:

- config rejects unknown fields;
- paths resolve correctly;
- stage dependencies validate.

## Milestone 2 — evaluator freeze

Deliver:

- canonical metric wrapper;
- Raw Direct exact reproduction;
- split definitions;
- minimal metric files.

Acceptance:

- S0 passes.

## Milestone 3 — operator library

Deliver:

- spatial;
- temporal;
- separable spatiotemporal;
- numerical diagnostics;
- synthetic tests.

Acceptance:

- S1A passes.

## Milestone 4 — QMC and fixed screen

Deliver:

- master designs;
- label-free anchor selection;
- S1 runner;
- decision summary.

Acceptance:

- 8-point smoke and full preflight succeed;
- no large run is launched automatically.

## Milestone 5 — global mixture

Deliver:

- S2 training;
- inner/outer split;
- minimal checkpointing;
- early stopping;
- controls.

Acceptance:

- tiny synthetic and one-fold smoke pass.

## Milestone 6 — conditional advanced stages

Implement S3 only after S2 evidence.

Implement S4 only after heterogeneity evidence.

This is a code-development dependency as well as a run dependency. Do not write the dense adaptive model preemptively if S1/S2 fail.

## Milestone 7 — ICA utility

Implement matched controls and run only the permitted ranks.

## Milestone 8 — finalist audit

Generate full scientific evidence only for the final promoted lane and its principal comparator.

---

# 26. Example high-level manifest

The exact schema may evolve, but the example should express:

```json
{
  "schema_version": 1,
  "experiment_id": "spon_ca_burst_learned_operator_selection_v1",
  "source_video": "../Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
  "labels_tsv": "../Inputs/Spon Ca Burst/labels/labels_normalized.tsv",
  "output_dir": "../Outputs/LearnedOperatorSelection/spon_ca_burst_learned_operator_selection_v1",
  "frames": {
    "review_start_ui": 1800,
    "review_end_ui": 2359,
    "quiet_start_ui": 1800,
    "quiet_end_ui": 1899,
    "frame_period_ms": 20.0
  },
  "evaluation": {
    "temporal_pool": "lme0.25",
    "nms_distance_px": 6,
    "match_radius_px": 6,
    "fixed_candidates_per_burst": 58,
    "quiet_false_peaks_per_map": 1.0
  },
  "design": {
    "sampler": "scrambled_sobol",
    "master_seed": 20260819,
    "master_size": 64,
    "allowed_prefix_sizes": [8, 16, 32, 64],
    "default_max_prefix_size": 32,
    "lhs_confirmation_size": 16
  },
  "model_seeds": {
    "screen": [7, 13, 19],
    "confirmation": [7, 13, 19, 29, 37]
  },
  "scientific_audit": {
    "enabled": false,
    "opt_out_reason": "User explicitly requested essential-metrics-only sequential screening on 2026-08-19. Full scientific audit is mandatory for promoted finalists."
  },
  "resources": {
    "device": "cuda",
    "cpu_threads": 4,
    "max_ram_mib": 16384,
    "min_free_disk_mib": 16384,
    "max_output_mib": 2048,
    "gpu_reserve_mib": 2048
  }
}
```

The strict config must include explicit operator bounds and stage gates rather than hiding them in code.

---

# 27. How Codex should react as results return

## After S0

- If exact: proceed to S1.
- If metric mismatch: diagnose evaluator only.
- Do not begin operator work on an invalid evaluator.

## After the first 8 S1 points

- If catastrophic and redundant: stop that family.
- If uncertain, diverse, or boundary-seeking: reveal 16.
- Do not jump directly to 64.

## After 16 S1 points

- If Fixed-Select promotes: include the family.
- If only the oracle is strong: mark complementary and include a few label-free anchors.
- If neither is useful: stop family.
- If condition number explains failures: consume one rescue.
- If well conditioned and poor: do not rescue.

## After S2 pilot

- Positive and reasonably consistent: run three seeds.
- Negative with strong oracle complementarity: diagnose calibration/optimization once.
- Negative without complementarity: stop.
- Do not build adaptive gating as a reflex.

## After S2 confirmation

- Strong gate: promote global mixture.
- Small gain: report as research-interest only.
- Tie: retain Fixed-Select or Raw Direct.
- Seed instability: compress bank once; otherwise stop.

## After S3

- Stable noninferior scale: promote continuous model.
- Underperformance: retain discrete mixture.
- Boundary solution: widen once.
- Unidentifiable: stop continuous scale.

## Before S4

- Calculate the heterogeneity evidence explicitly.
- If the entry gate is not met, write `not_justified` and skip S4.

## After S4

- Require improvement over global.
- Require hidden-identity success.
- If tied, keep global.

## After S5

- Compare against no-rotation and random rotation.
- If ICA does not add at least the practical threshold, stop treating ICA as the central method.
- If pairwise ICA aligns with difference, document it as a temporal derivative.
- If ICA promotes, retain only the winning rank/objective and complete the full audit.

---

# 28. Final report structure

The program-level final report should answer, in order:

1. Did the evaluator reproduce?
2. Which operator families were numerically valid?
3. Did any fixed whitening operator beat Raw Direct leakage-safely?
4. Was there a meaningful oracle/complementarity gap?
5. Did global learned mixing beat Fixed-Select?
6. Could continuous scale replace the bank?
7. Was adaptivity justified and useful?
8. Did ICA beat matched non-ICA and random-rotation controls?
9. What failed because of conditioning?
10. What failed despite good conditioning?
11. What method, if any, should be retained?
12. What claims remain impossible because only one video is available?

Use explicit outcome language:

```text
supported
provisionally supported
not supported
numerically unresolved
not tested because prerequisite failed
```

Do not substitute “completed” for “supported.”

---

# 29. Potential alternatives — documented, not implemented in this program

If learned whitening and ICA fail, do not immediately create a larger whitening/ICA search. Preserve the results and consider the following future branches.

## 29.1 Predictive residual and latent dynamics

The repository’s offline latent smoother has already shown stronger known-positive recovery than Raw Direct under one operating contract, though confirmation is incomplete.

Future question:

> Is predictable background removal a better assumption than statistical independence?

Possible direction:

\[
X_t = \widehat X_t^{\mathrm{predictable}} + R_t.
\]

Then evaluate the residual under the same canonical metric and preservation contract.

## 29.2 Low-rank plus sparse decomposition

Model:

\[
X = L + S + N,
\]

with:

- \(L\): slowly varying or low-rank background;
- \(S\): sparse event activity;
- \(N\): residual noise.

This is a strong theoretical baseline but requires careful identifiability and must not assume that all neural signal is sparse or that background is globally low rank.

## 29.3 Structured NMF

Use nonnegativity and temporal persistence rather than independence.

Useful when additive positive sources are more plausible than statistically independent signed components.

## 29.4 Self-supervised predictive masking

Predict masked pixels/frames or future local context, then use the unpredictable residual as event evidence.

This may learn scale without direct label tuning but requires explicit controls against learning the events themselves as predictable background.

## 29.5 Information-theoretic regularization without full ICA

Learn a task-relevant representation while adding a bounded dependence penalty such as:

- CS-QMI;
- HSIC;
- InfoMax;
- entropy-rate terms.

This asks whether partial independence helps without forcing complete ICA separation.

## 29.6 Carrier-preserving cross-scale coherence

Existing repository evidence suggests that coherence-like features may improve compact candidate recovery in some settings.

A future branch could learn cross-scale consistency while retaining Raw Direct as the carrier.

## 29.7 Hierarchical residual separation

Use a residual architecture:

\[
Z^{(\ell+1)}
=
Z^{(\ell)}
+
\beta_\ell O_{\theta_\ell}(Z^{(\ell)}),
\]

with each stage able to turn itself off.

This is preferable to repeatedly destructive whitening, but it is not justified until the single-stage operator program demonstrates complementary scales.

## 29.8 Learned sparse dictionary or convolutional sparse coding

Use localized atoms and sparse coefficients rather than independent global components.

This may match spatially localized calcium events better than global ICA, but it adds substantial identifiability and optimization questions.

These alternatives should be ranked after the current program based on the observed failure mode:

- no fixed whitening utility;
- useful whitening but no learned advantage;
- useful representation but no ICA advantage;
- conditioning-limited spatiotemporal model;
- within-video overfitting.

---

# 30. Final acceptance checklist

## Evaluator

- [ ] Raw Direct Q1 metric reproduces exactly.
- [ ] Raw Direct fixed-budget metric reproduces.
- [ ] Candidate ordering is deterministic.
- [ ] Frame and coordinate overlay passes.
- [ ] Macro and pooled metrics are distinct.
- [ ] Unknown candidates are not called false positives.

## Data discipline

- [ ] Outer held-out bursts are inaccessible during training.
- [ ] Inner selection uses only outer-training bursts.
- [ ] Absolute coordinates are not model inputs.
- [ ] Hidden-identity confirmation exists for finalists.
- [ ] Current-video limitation is stated.

## QMC design

- [ ] Master Sobol designs are generated before evaluation.
- [ ] Prefixes are nested powers of two.
- [ ] Quantization and duplicates are deterministic.
- [ ] Categorical families are separate.
- [ ] Anchor selection is label-free.
- [ ] LHS is confirmation-only.
- [ ] No false claim of variance-based Sobol sensitivity is made.

## Operators

- [ ] Fractional whitening is tested.
- [ ] Shrinkage and eigen flooring are explicit.
- [ ] Raw-preserving blend includes exact Raw Direct.
- [ ] Spatial and temporal alignment are correct.
- [ ] Separable spatiotemporal whitening is the default.
- [ ] Full joint whitening is guarded.
- [ ] Outputs are quiet-standardized before mixing.

## Sequential execution

- [ ] No automatic all-stage command exists.
- [ ] Every stage writes `decision.json`.
- [ ] Next stages refuse invalid prerequisites.
- [ ] One diagnostic rescue maximum per branch.
- [ ] Stop decisions are preserved.
- [ ] Output roots never collide or overwrite.

## Metrics and storage

- [ ] Only essential run metrics are retained during screens.
- [ ] No full media are generated for every cell.
- [ ] Compact parameters are reproducible.
- [ ] Runtime and peak memory are recorded.
- [ ] Finalists restore the full scientific audit.

## Learned mixture

- [ ] Fixed-Select is the main comparator.
- [ ] Uniform and random mixes are controls.
- [ ] Raw Direct is an anchor.
- [ ] Model capacity is bounded.
- [ ] Five-seed promotion gate is enforced.

## Continuous/adaptive stages

- [ ] Continuous scale runs only after evidence.
- [ ] Boundary expansion occurs at most once.
- [ ] Adaptivity requires heterogeneity evidence.
- [ ] Adaptive model must beat the global model.
- [ ] Hidden-identity evaluation is mandatory for adaptivity.

## ICA

- [ ] No rank-64 initial experiment.
- [ ] Matched no-rotation control exists.
- [ ] Random orthogonal rotation control exists.
- [ ] Same rank/evidence/budget is used.
- [ ] Convergence and component stability are reported.
- [ ] Pairwise derivative alignment is classified correctly.
- [ ] ICA advances only on incremental biological utility.

## Final interpretation

- [ ] Numerical failure is distinguished from scientific failure.
- [ ] Completion is not called success.
- [ ] Within-video evidence is not called cross-fish generalization.
- [ ] Potential alternatives are documented but not silently implemented.
- [ ] The retained method is the simplest method that passes its gate.

---

## 31. Default immediate next action

Codex should begin with **Milestones 1 and 2 only**:

1. add the workflow/config/state-machine skeleton;
2. wrap the existing evaluator;
3. reproduce Raw Direct exactly;
4. implement metric-only screening artifacts;
5. write the S0 decision;
6. stop and report before implementing or running the full operator screen.

This ensures that every later result rests on a frozen, reproducible evaluation contract.
