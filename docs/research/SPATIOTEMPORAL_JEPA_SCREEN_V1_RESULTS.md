# Compact spatiotemporal JEPA screen v1 results

## Technical summary

The bounded one-seed JEPA screen is a **negative escalation result for this
architecture, objective, score head, and 500-step budget**. Under the common
frozen latent-temporal-change head, JEPA recovered 1.85% of exact injected
sources at the registered up-to-four-proposal operating point, versus 18.75%
for the frozen handcrafted carrier/context/kinetic stack. The paired difference
was -16.90 percentage points, with a recording/window/injection-seed
hierarchical 95% bootstrap interval of -31.71 to -3.70 percentage points.

The representation also crossed both registered collapse thresholds at this
screen budget: its effective rank was 3.29 of 64 dimensions and its first
principal component explained 73.70% of validation-token variance. Because the
registered estimand requires three seeds at 5,000 updates, this is a screen-level
threshold violation rather than formal failure of the experiment-level gate.
Training loss therefore cannot be treated as evidence that the learned
representation captured local neuron-like events.

The result does **not** complete `NREV-EXP-0028`. The screen evaluated one of
three registered seeds and 500 of 5,000 planned optimizer steps, did not satisfy
the motion dependency, and did not produce the required scientific-audit
media. It supports a project decision to **not escalate JEPA v1 automatically**,
while leaving the broader JEPA family unresolved.

## The interpretable stack remained the strongest primary method

The primary metric is the mean source-on recall across 108 paired
empirical-background fixtures. Each 32-by-64-by-64 fixture contains one, two,
or four exact injected sources. A method may return up to four separated local
maxima; fewer maxima are allowed and recorded. Source counts are nested within
36 recording/window/injection-seed clusters rather than treated as independent
replicates.

| Primary method | Macro source-on recall | Exact sources recovered | Mean proposals per fixture | Recovered sources per proposal |
| --- | ---: | ---: | ---: | ---: |
| Compact JEPA, latent temporal change | 1.85% | 5 / 252 | 2.25 | 2.06% |
| Masked pixel autoencoder, latent temporal change | 1.39% | 5 / 252 | 3.40 | 1.36% |
| Frozen random encoder, latent temporal change | 3.01% | 9 / 252 | 3.15 | 2.65% |
| Frozen handcrafted carrier/context/kinetic stack | **18.75%** | **45 / 252** | 4.00 | **10.42%** |

The proposal-count difference explains only part of the gap. JEPA returned 243
proposals across the 108 fixtures, compared with 432 for the handcrafted arm,
but its recovered-source yield per proposal was still about five times lower.
The frozen random encoder also exceeded the trained JEPA under the same score
head. At this screen budget, masked latent prediction did not add useful local
event evidence.

The strongest comparator was reselected inside every bootstrap draw. The
handcrafted arm was selected in 99.7% of 1,000 draws; the masked autoencoder was
selected in 0.3%, and the random encoder in none. The observed JEPA-minus-
strongest effect was -0.16898 with 36 top-level clusters and four recording
groups.

The maintained configuration registered 5,000 bootstrap draws, but Screen B
stored 1,000 and did not serialize that reduction as an execution override.
This is a real provenance mismatch. A read-only recomputation from the frozen
table using all 5,000 registered draws produced the same conclusion: interval
-31.71 to -3.70 percentage points, with the handcrafted comparator selected in
99.74% of draws. The corrected recomputation is regression-tested, but it does
not retroactively make the run claim-bearing.

## The negative effect was not confined to one recording or fixture type

| Held empirical background | JEPA recall | Handcrafted recall | JEPA minus handcrafted |
| --- | ---: | ---: | ---: |
| `060126_10_rest` | 1.85% | 34.26% | -32.41 pp |
| `060126_12_left` | 1.85% | 25.00% | -23.15 pp |
| `060126_15_right` | 0.00% | 10.19% | -10.19 pp |
| Spon Ca Burst quiet windows | 3.70% | 5.56% | -1.85 pp |

The direction was negative in all four backgrounds. It also survived the main
descriptive cuts:

- JEPA recovered 0 of 36 isolated single-source injections, while the
  handcrafted arm recovered 8 of 36.
- JEPA macro recall was 2.08%, 1.39%, and 2.08% in low-, median-, and
  high-temporal-MAD windows; the handcrafted values were 27.08%, 15.28%, and
  13.89%.
- JEPA recovered 4 of 128 crescent sources and 1 of 124 ellipse sources. The
  handcrafted arm recovered 24 of 128 and 21 of 124, respectively.
- Using continuous placement centers instead of footprint peaks changed JEPA's
  macro recall only from 1.85% to 1.62%. Center semantics do not explain the
  primary failure.

These are descriptive sensitivities, not additional independent hypothesis
tests. Their value is diagnostic: the aggregate result is not driven solely by
crowding, morphology, amplitude stratum, or one held recording.

## The intervention maps reveal background competition, not a rescued endpoint

The source-on map is the registered deployment-like endpoint. The intervention
map subtracts the source-off score map from the source-on map and asks whether
the injected perturbation changed the method's evidence at the correct
location. It is a mechanism-only diagnostic because a real deployment does not
have a matched source-off counterfactual.

| Score lane | Source-on macro recall | Intervention-map macro recall |
| --- | ---: | ---: |
| JEPA latent temporal change | 1.85% | 25.69% |
| JEPA masked-prediction error, secondary | 0.23% | 20.83% |
| MAE latent temporal change | 1.39% | 14.81% |
| MAE reconstruction error, secondary | 8.33% | 92.59% |
| Frozen random latent temporal change | 3.01% | 20.14% |
| Frozen handcrafted stack | 18.75% | 85.19% |

The MAE reconstruction-error and handcrafted intervention maps respond strongly
to the injected signal, yet many injected sources are not among the strongest
absolute source-on maxima. This identifies a difficult background-ranking
problem. It does not rescue the JEPA primary result: JEPA is weak both in the
source-on endpoint and in the intervention diagnostic, and objective-native
JEPA and MAE errors are not directly comparable scientific measurements.

## Optimization succeeded mechanically but the representations crossed the screen thresholds

The GPU screen used bfloat16 autocast with observed float32 JEPA and MAE loss
tensors. JEPA masked-latent loss decreased from 1.7461 at step 1 to 0.00946 at
step 500; validation loss was 0.01460. The representation nevertheless became
highly concentrated.

| Frozen representation | Effective rank | Rank fraction | PC1 variance share | Screen threshold status |
| --- | ---: | ---: | ---: | --- |
| JEPA | 3.29 / 64 | 5.15% | 73.70% | Outside both thresholds |
| Masked pixel autoencoder | 1.69 / 64 | 2.64% | 90.39% | Outside both thresholds |
| Random encoder | 6.42 / 64 | 10.03% | 59.69% | Outside PC1 threshold |

The random encoder's failure on the PC1 threshold warns that the diagnostic is
strict for this video domain. Even with that caveat, JEPA's effective rank is
about half the random encoder's and its common-head recovery is worse. A likely
interpretation is that the objective learned a small number of predictable
global or acquisition modes instead of a spatially discriminative event
representation. This is a hypothesis, not a causal conclusion.

The MAE training curve was also unstable: the recorded pre-clipping gradient
norm reached 723.93 at step 500 despite a configured clip norm of 1.0. This does
not invalidate the matched screen, but it is another reason not to promote the
deep arms from training loss alone.

## Scope, data, and validation

- **Self-supervised corpus:** 11 hash-verified raw `060126` TIFF recordings,
  17,664 frames at 764 by 1,046 pixels. Eight complete recordings were used for
  training and three for validation. Spon was excluded from self-supervision
  and used only for post-training empirical-background injection.
- **Training sample:** one registered training seed (`1001`), 512 unique raw
  clips, and the shared seed-`2001` validation bank of 96 clips. Training-only
  robust normalization had center 489.0 and scale 289.107 native units.
- **Exact-truth evaluation:** 4 backgrounds by 3 disjoint windows by 3
  injection seeds by 3 source counts = 108 fixtures and 252 injected sources.
  The complete six-method table contains 648 unique method-fixture rows with no
  missing or duplicate keys.
- **Pair closure:** all source-on pairs closed within 0.5 float32 ULP. The
  maximum absolute residual, 0.0002431 native units, is reported descriptively
  rather than used as a scale-independent threshold.
- **Artifact integrity:** all 25 entries in `artifact_index.json` recomputed to
  their recorded SHA-256 and byte count, totaling 389,292,601 bytes. No
  workstation path was found in the text outputs.
- **Motion screen:** 10 of 11 recordings were clear under the bounded heuristic
  screen. `060126_06_left` crossed the tile-disagreement review threshold
  (95th percentile 1.643 pixels versus 1.5). This is not the registered
  `NREV-EXP-0025` motion-residual analysis and does not satisfy that dependency.
- **Nuisance analysis:** the available probes are descriptive. Nuisance
  residualization, counterfactuals, leave-one-recording-out primary analysis,
  per-recording collapse diagnostics, and cross-seed CKA remain unrun.
- **Secondary-panel completeness:** duplicate/weak-neighbor recovery and the
  registered source-pixel AUC against displaced controls were not emitted.
  Existing intervention and morphology sensitivities do not replace them.

The first attempted screen is preserved as
`NREV-RUN-EXP-0028-SCREEN-20260829-A.partial`. It failed safely when an absolute
pair-closure threshold treated normal float32 rounding as an error. The
contract was corrected to the prespecified 0.51-ULP criterion, the failed
partial was retained, and the successful B run used a new non-colliding output
root.

## Evidence boundary

This screen establishes only that the current JEPA v1 configuration should not
advance automatically from a bounded engineering screen. It does not establish
that:

- JEPA as a model family is unsuitable for calcium imaging;
- 5,000-step, three-seed training would necessarily reproduce the same effect;
- the injected sources exhaustively represent biological neurons;
- unmatched proposals are false positives;
- the recordings are independent animals;
- the representation is free of motion or acquisition confounding; or
- the model performs reinforcement learning, action-conditioned prediction, or
  control.

The result is therefore **shareable with these caveats**, but it is not a
claim-bearing completion or a biological-identification result.

## Decision and next automated work

1. **Keep JEPA v1 on hold and do not spend the remaining 29,000 matched
   optimizer steps automatically.** The unfavorable common-head estimate and
   screen-level collapse-threshold violations support a conservative post-screen
   engineering hold. They are not prespecified futility stopping rules and do
   not formally resolve the experiment-level gates.
2. **Run `NREV-EXP-0025` next.** Estimate global and tile motion, registration
   residuals, and their relation to trace and feature stability. The current
   heuristic flag is useful triage, not a substitute.
3. **Advance `NREV-EXP-0021` with the frozen interpretable stack.** The
   handcrafted carrier/context/kinetic features remain the strongest current
   input for grouped positive-unlabeled and label-policy sensitivity tests.
4. **Complete a full-field acquisition and normalization audit.** Map noise,
   striping, dtype-rail occupancy, local-standardization floor use, and quiet
   null stationarity rather than extrapolating center-only checks.
5. **Revisit JEPA only as a new versioned hypothesis.** A JEPA v2 should first
   demonstrate non-collapse and beat the frozen random encoder on a held-
   recording screen. Any anti-collapse regularizer, nuisance-residual target,
   local/global factorization, or new score head changes the scientific object
   and requires a new configuration and run ID.

## Further questions

- Does motion/background residualization improve the existing handcrafted
  source-on operating point without erasing real event structure?
- Why does MAE reconstruction error recover 92.59% on intervention maps but
  only 8.33% on source-on maps? A label-free rank decomposition could quantify
  how often native background maxima displace injected sources.
- Is the low-dimensional embedding a property of the video domain, the encoder
  architecture, or the masked-prediction objective? The random-encoder PC1
  result makes this an important control question for any v2.
- Can the interpretable stack's source-off calibration be converted into a
  stable feature family for uncertainty-aware learning without using exact
  injection outcomes during fitting?

## Reproducibility record

- Protocol: [compact spatiotemporal JEPA representation pilot v1](../workflows/spatiotemporal_jepa_representation_v1.md)
- Experiment: `NREV-EXP-0028`
- Successful bounded screen: `NREV-RUN-EXP-0028-SCREEN-20260829-B`
- Runtime status: `screen_complete_claim_gates_unresolved`
- Execution window: 2026-08-30 03:26:19--03:26:40 UTC
- Registered config SHA-256:
  `5a2a0a8fc174dc770165deb1c634552800223ba91e96f8158fe5a451ff630679`
- Artifact-index SHA-256:
  `6e6c7ef9d8c70c2c52fcb809ea1f4a65578987e558302a3a63b576cb9eb448bd`
- Summary SHA-256:
  `0d89907e2e730cc0af0e5013a7615b5e4d4d0fecc5952830e05e4aa306dec068`
- Paired-result JSON SHA-256:
  `950fc024611c945a1b5f3b88f57f6cba1b4631c0bd7503f4004334f308fc5617`

The local run package lives under
`Outputs/NeuronIdentifiability/NREV-EXP-0028/runs/`. It is intentionally not a
Git-tracked substitute for the portable protocol, registry, and bounded result
record.
