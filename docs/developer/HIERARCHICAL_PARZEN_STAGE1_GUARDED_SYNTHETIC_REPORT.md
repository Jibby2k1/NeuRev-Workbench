# Guarded Stage-1 multi-seed synthetic report

Last updated: 2026-07-29.

## Decision

The recursive safety architecture works, but the learned separation methods do
not yet pass the scientific gate.

- **Numerical-stability gate: pass.**
- **Scientific-validity gate: fail.**
- Do not advance to Stage 2 or real Spon execution.

This experiment used generated arrays only. It did not load Spon Ca Burst,
labels, TIFFs, or a GPU model.

## Exact experiment

Output:

```text
Outputs/HierarchicalParzenICA/stage1_guarded_synthetic_multiseed_v1
```

The matrix contains:

```text
12 cases * 5 seeds * 4 methods = 240 combinations
```

All 240 combinations completed. The output contains 240 progress records and
240 result rows. The CPU-only run used two numerical threads, took 1.69
seconds, and produced 532 KB of artifacts.

### Cases

1. static background plus white noise;
2. multiplicative gain drift;
3. linear background drift;
4. nonlinear slow drift;
5. fast localized event;
6. slow ramp and plateau;
7. background and signal with similar persistence;
8. pure noise;
9. one-pixel translation edge;
10. signal with saturation/clipping;
11. heteroscedastic noise; and
12. equal-staticness unresolved challenge.

Seeds were `7`, `13`, `19`, `29`, and `37`. Every movie had 20 frames of
24-by-24 pixels, an eight-frame calibration prefix, lag one, 512 bounded fit
samples, ordinary covariance, and no labels in fitting.

### Learned-method controls

Batch CS-Parzen used bandwidth 0.35, 64-row kernel blocks, a 15-degree screen,
and a two-degree refinement window with one-degree steps.

Stochastic Parzen used:

- gain-aware reference initialization;
- learning rate `0.0002`;
- maximum update angle `0.25` degrees;
- gradient clip `5`;
- dictionary size `32`;
- dictionary warmup `128`;
- batch size `128`;
- at most 25 iterations; and
- convergence tolerance `1e-5`.

All 60 stochastic fits met that numerical convergence criterion.

### Recursive safety contract

For

```text
B(t) = a * B(t-1) + b * I(t) + offset
```

the applied model required:

| Bound | Value |
| --- | ---: |
| Maximum `abs(a)` | 1.2 |
| Maximum `abs(b)` | 0.1 |
| Maximum reconstruction-operator norm | 2.0 |
| Maximum learned fraction | 0.1 |
| Minimum tested learned fraction | 0.0015625 |
| Nonconverged policy | Adaptive-reference fallback |

The learned demixer was aligned to the reference and tested at successively
halved learned fractions. The largest safe fraction was applied. A fit with no
safe learned fraction used the reference exactly.

## Numerical outcome

The guard eliminated the earlier catastrophic batch recursion:

- 240/240 runs completed;
- no applied feedback operator violated the declared bounds;
- maximum saved-output closure error was below `5.98e-08`;
- no output-scale explosion occurred; and
- the numerical-stability gate passed.

Raw learned feedback was nevertheless unsafe frequently:

| Method | Raw feedback rejected | Exact-reference anchoring |
| --- | ---: | ---: |
| Batch CS-Parzen | 57/60 | 7/60 |
| Stochastic Parzen | 21/60 | 0/60 |

Batch accepted learned fractions were usually extremely small: 19 fits used
`0.0015625`, 18 used `0.003125`, and seven used zero. Stochastic anchoring was
less severe: 55/60 fits retained the maximum 0.1 learned fraction.

This is a safety success, not evidence that the learned method is useful.

## Scientific outcome

Signal NMSE is measured after subtracting the known synthetic artifact and
measurement-noise arrays from the residual. Lower is better; zero is ideal.

| Method | Median signal NMSE | Worst signal NMSE | Median background NMSE | Worst background NMSE |
| --- | ---: | ---: | ---: | ---: |
| Fixed reference | 0.03056 | 0.30749 | 0.000760 | 0.01599 |
| Adaptive reference | **0.03039** | 0.30931 | **0.000522** | 0.01863 |
| Guarded batch CS-Parzen | 0.15365 | 2.31284 | 0.001748 | 0.02773 |
| Guarded stochastic Parzen | 0.05687 | **0.19526** | 0.000911 | **0.01134** |

The adaptive reference remains the best general default. The stochastic lane
has a better worst-case signal error and background worst case, but its median
signal error is 1.87 times the adaptive reference.

### Matched signal comparisons against adaptive

There were 20 matched signal-bearing case/seed comparisons:

| Method | Wins | Ties | Losses | Median NMSE ratio to adaptive |
| --- | ---: | ---: | ---: | ---: |
| Fixed reference | 5 | 0 | 15 | 1.003 |
| Guarded batch CS-Parzen | 2 | 3 | 15 | 7.718 |
| Guarded stochastic Parzen | 5 | 0 | 15 | 1.988 |

Every stochastic win occurred in the `similar_persistence` challenge, for all
five seeds. Its median NMSE there was `0.18756`, compared with `0.29029` for
adaptive. This is the clearest positive result: constrained stochastic Parzen
may help when the background and neural signal evolve on similar timescales.

It did not generalize to the easier event cases:

| Case | Fixed | Adaptive | Batch | Stochastic |
| --- | ---: | ---: | ---: | ---: |
| Fast event | 0.04728 | **0.04682** | 0.08784 | 0.05157 |
| Slow ramp/plateau | 0.00841 | **0.00833** | 0.15495 | 0.03498 |
| Similar persistence | 0.28758 | 0.29029 | 1.22168 | **0.18756** |
| Saturation/clipping | 0.00946 | **0.00937** | 0.11778 | 0.02895 |

### Artifact behavior

For a one-pixel translation, amplitude gain near one means the artifact remains
in the residual for later artifact handling. Adaptive retained essentially all
of it (`0.99999`). Stochastic retained `0.803`, while batch retained only
`0.629`; the learned lanes therefore absorbed part of the motion edge into
background.

For saturation/clipping, adaptive preserved neural amplitude (`1.008`) and
artifact amplitude (`0.976`) closely. Stochastic reduced neural amplitude to
`0.857` and amplified artifact amplitude to `1.50`. Batch reduced neural
amplitude to `0.674` and amplified artifact amplitude to `2.14`.

### Unresolved behavior

All four methods resolved all five equal-staticness challenges. The expected
result was `unresolved`. The staticness selector is therefore overconfident,
and the unresolved gate failed `0/5` for every method.

## Interpretation

1. The safety guard is effective and should remain mandatory.
2. Raw batch CS-Parzen is structurally incompatible with this recursive
   background role in most tested cases; a wider batch hyperparameter sweep is
   not justified.
3. Reference-initialized stochastic Parzen is the only learned lane with a
   repeatable scientific advantage, but only for similar-persistence sources.
4. That advantage is not currently selectable from quiet calibration alone and
   is accompanied by worse fast-event, plateau, clipping, and motion behavior.
5. Staticness confidence is not a sufficient unresolved or neural-confidence
   measure.

## Next justified implementation

The next checkpoint should focus narrowly on the stochastic lane:

1. add an explicit penalty on current-observation leakage `abs(b)`;
2. add motion-edge and saturation preservation constraints;
3. redesign unresolved classification using blockwise score stability and an
   absolute evidence threshold, not only a two-component margin;
4. preregister a small learned-fraction and regularization experiment on these
   generated cases;
5. require improvement over adaptive across event classes, not merely overall
   worst case; and
6. integrate accepted-model sign/permutation tracking.

Only after those conditions improve across seeds should semi-synthetic
quiet-Spon injection be considered.
