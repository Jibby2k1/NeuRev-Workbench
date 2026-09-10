# NeuRev research story

Generated from the repository research registry; do not edit by hand.

NeuRev turns difficult neural-imaging movies into auditable evidence: it builds interpretable measurements, proposes candidate activity, exposes failure modes, supports blinded review, and records which conclusions have or have not survived validation.

## Programs

| Program | Lifecycle | Registered scope |
| --- | --- | --- |
| Neuron Identifiability | active | An intensive within-recording methods study of 106 confirmed occurrences at 50 immutable sites in one calcium-imaging recording. |
| Source Separation and Representation | active | Foundational representation and source-separation methods used by the repository's imaging workflows. |
| Fish Intent and Inverse Control | draft | A downstream causal program that may proceed only after measurement, leakage, intervention, and safety gates are satisfied. |
| Grid and Latent Dynamics | active | Measurement and forecasting methods for template-aligned neural state, without causal or control interpretation. |
| Gamma Local Standardization | active | A distinct operator and deployment-characterization campaign: 79 protected Spon occurrences from 26 identities, an annotation-free full-record proposal ledger, one independent 15 right sparse-positive readout, and exact-truth simulations. |

## Registry snapshot

- 20 atomic claims
- 19 completed historical experiments with portable evidence capsules
- 13 planned experiments
- 9 registered run records

Historical v1 results intentionally have no fabricated run records; their capsules preserve results, hashes, and explicit provenance gaps. Post-migration executions of drafted v1 entries are registered explicitly and remain separate from scientific completion.

## Current flagship boundary

An intensive within-recording methods study of 106 confirmed occurrences at 50 immutable sites in one calcium-imaging recording.

Unresolved boundaries:

- Full-field precision is unresolved.
- One-to-one biological source identity is unresolved.
- Generalization to an independent recording is unresolved.

## Documented results beyond the historical capsules

These report-backed records preserve computed outcomes and decisions. They do not add portable evidence capsules or promote the flagship claims. Different programs, populations, and evaluation heads remain separate.

### NREV-EXP-0021: Uncertainty-Aware Positive-Unlabeled Feature Fusion V1 with v1.1 output-assembly amendment

Program `NREV-PRG-0001`; registry lifecycle **draft**, outcome **not_evaluated**, evidence tier **none**; action **hold**.

Run B completed 1,619 OOF candidate scores across 78 positives and 1,541 unknown U candidates. Macro-fold SPU-AUC was 0.9740218518 for tiny MLP, 0.9740959140 for linear, 0.9756013670 for elastic, 0.9754961258 for bagged PU, 0.9501444137 for equal-weight expert fusion, 0.9335176617 for carrier, and 0.7646602748 for same-union CFAR reranking. MLP-minus-linear was -0.0000740622 with interval [-0.0046074197, 0.0059124340]; MLP-minus-CFAR was +0.2093615770 with interval [0.1207303370, 0.3036814877]. MLP and bagged PU each recovered 34 of 78 at budget 58, versus 33 linear, 30 expert, 29 carrier, and 14 CFAR. Seed stability and component association passed, but nonlinear advance and the required burst/review-policy sensitivity panel did not. Scientific audit and promotion remain incomplete.

[Result and provenance](../../docs/research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0024.yaml)

**Boundary:** Preserve NREV-RUN-EXP-0021-SCREEN-20260830-A and its `.partial` tree as failed, immutable, and nonresumable; do not relabel it succeeded or use its unserialized model evaluation as result evidence. Preserve completed Run B, its frozen runner/config/protocol, and its exact output index; engineering PASS is separate from the failed nonlinear advance signal, incomplete scientific audit, and false claim-promotion state. Do not tune MLP width, depth, optimizer, seed selection, feature set, folds, review reserve, or favorable subgroups on Run-B outcomes and call the result confirmation. Keep U candidates unknown rather than negative, quiet candidates source-off controls rather than negatives, and later review labels locked and descriptive. Treat cfar_score only as same-union scalar reranking and one fusion input; do not describe the favorable MLP-CFAR contrast as end-to-end CFAR replacement or nonlinear causal benefit. Do not report ordinary precision, specificity, false-positive rate, calibrated neuron probability, causal feature importance, independent-recording generalization, biological identity, or scientific completion from this screen. Whole-component intervals are descriptive because positive-containing component support is only 9, 3, 3, 2, and 4 across folds; stable MLP seeds do not replace independent biological replication.

### NREV-EXP-0031: Contextual-envelope feature semantics pilot v1

Program `NREV-PRG-0001`; registry lifecycle **draft**, outcome **not_evaluated**, evidence tier **none**; action **retain**.

The pilot completed 360 metric rows across three sources, five temporal windows, three spatial supports, and eight feature forms; all 18 representative videos decoded. Added spatial support inflated diffuse and quiet evidence, while H=A^2/U was the cleanest attenuation layer. This is engineering output, not a scientific-audit-complete detector experiment.

[Result and provenance](../../docs/research/CONTEXTUAL_ENVELOPE_FEATURES_V1_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0029.yaml)

**Boundary:** No detector, candidates, full-field precision, biological specificity, identity, or independent transfer was established. The apparent A*U contrast partly reflects the no-pooling A-squared nonlinearity; local coherence retains its separate existing definition.

### NREV-EXP-0032: Contextual-envelope retrieval comparison v1

Program `NREV-PRG-0001`; registry lifecycle **draft**, outcome **not_evaluated**, evidence tier **none**; action **hold**.

All 106 occurrences at 50 sites produced 2226 occurrence-feature rows and 106 trace comparisons. No aligned attenuation arm passed the gate against both A-squared and its matched misalignment control. Temporal peak localization nearly tied the amplitude control; the 3-by-3 spatial envelope reduced localization. The run report records validation passed and scientific promotion false.

[Result and provenance](../../docs/research/CONTEXTUAL_ENVELOPE_RETRIEVAL_V1_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0030.yaml)

**Boundary:** This is within-recording operator retrieval; no biological negatives, full-field precision, specificity, identity, or transfer were established. The existing registry retains its draft/evidence-none status until normalized run and audit provenance is reconciled; this decision makes the completed report visible without promotion.

### NREV-EXP-0033: Temporal contextual-envelope morphology audit v1

Program `NREV-PRG-0001`; registry lifecycle **validated**, outcome **supported**, evidence tier **current_recording**; action **retain**.

The validated current-recording morphology run evaluated six operators across three sources over 106 occurrences at 50 sites with 2000 site-bootstrap resamples. H=A^2/U preserved the aligned anchor, increased event-area/core concentration, and reduced post-peak shoulder across all three sources with paired intervals excluding zero. It produced 106 occurrence panels and three fixed-scale 560-frame videos. Scientific promotion remains false.

[Result and provenance](../../docs/research/CONTEXTUAL_ENVELOPE_MORPHOLOGY_V1_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0031.yaml)

**Boundary:** This validated operator audit created no detector, candidate coordinates, model annotations, or portable evidence capsule. Morphology changes do not establish better detection, specificity, biological waveform fidelity, identity, or independent transfer. Raw A remains necessary for biological waveform analysis because H deliberately alters decay morphology.

### NREV-EXP-0034: Gamma-LS protected, operational, independent, and timing campaign

Program `NREV-PRG-0005`; registry lifecycle **computed**, outcome **mixed**, evidence tier **descriptive**; action **retain**.

The h15 context won all four coordinate-free training folds but failed protected support sufficiency. Two-frame ICA did not pass its paired gain gate; radial superiority was not established. The q1 full-record ledger contains 371 frame-local proposals on 295 of 2259 frames. Its final model-only audit passed with 24 label-free surrogates, 25 decoded videos, 27 PNGs, no annotation source opened, and manual visual QA. The independent 15 right B58-per-block result was 6/51. Corrected batch throughput was 1470 to 2216 frames/s, while both strict 1-kHz paced trials failed.

[Result and provenance](../../docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) · [Decision](../registry/decisions/NREV-DEC-0025.yaml)

**Boundary:** This separate Gamma-LS program does not update the 106-occurrence identity-safe flagship claim ledger or its independent-generalization gate. The protected burst-occupancy, independent block-ranking, and operational frame-level proposal heads are different estimands. h15 is operational, not sufficient or optimal; 371 proposal rows are not unique neurons, events, or confirmed detections. One independent sparse-positive recording does not establish population generalization or biological precision, specificity, false-positive rate, or identity. Batch throughput does not establish zero-miss/zero-drop sustained 1-kHz readiness.

### NREV-EXP-0035: ICA and whitening synthetic-to-real factorial evaluation

Program `NREV-PRG-0002`; registry lifecycle **computed**, outcome **mixed**, evidence tier **descriptive**; action **retain**.

The real-data completion audit passed all eight gates: 30891 exact eligible fits, 28928 converged, 26333 whitening-resolved. Protected macro known-positive recall at B58 was 0.300 with burst-bootstrap interval 0.092 to 0.509. One finalist passed scientific-audit inventory, decode, and visual inspection. Its independent 15 right readout recovered 20/51 at B58 per block for each of three seeds from 18689 frozen candidates. The synthetic screen had 106080 fits and zero fits passing every original source-recovery gate; that failure was explicitly amended to a nonblocking warning.

[Result and provenance](../../docs/research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0026.yaml)

**Boundary:** No portable evidence capsule or new repository claim promotion is created by this retrospective report registration. The single independent recording and seed refits do not establish population-level generalization; unmatched proposals remain unknown. The ICA 20/51 result and Gamma-LS 6/51 result use different pipelines and candidate universes and are not a matched superiority test. Synthetic recovery failure remains a warning; stable ICA components do not establish biological source identity or biological precision.

### NREV-EXP-0036: PC-MITL-ICA paired development and locked specialized confirmation

Program `NREV-PRG-0002`; registry lifecycle **computed**, outcome **rejected**, evidence tier **computational_simulation**; action **stop**.

Across 80 development pairs, matrix-TC minus CS-Parzen mean held-out absolute source correlation was +0.00193 with interval [-0.01120, +0.01287]. Locked specialized confirmation produced +0.019481 across 40 primary pairs, 35/40 wins, and seed-cluster interval [+0.011270, +0.029030]. The sparse-calcium guardrail passed at -0.002331, but the primary mean missed the predeclared +0.020 threshold by 0.000519.

[Result and provenance](../../docs/research/PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0027.yaml)

**Boundary:** These are paired synthetic-fixture results; no real video or biological labels were used. Do not round, relax, or replace the predeclared +0.020 threshold after seeing the result. The specialized simulator result does not authorize trajectory matrix-TC, process-conditional objectives, or real-video escalation.

### NREV-EXP-0037: Feature Atlas v1 grouped incremental-reranking evaluation

Program `NREV-PRG-0001`; registry lifecycle **computed**, outcome **inconclusive**, evidence tier **descriptive**; action **hold**.

On the same 1619-candidate union, augmented linear macro held-fold SPU-AUC was 0.984392 versus 0.979734 for the retrained existing model, with 35/78 versus 34/78 at budget 58. The paired global SPU-AUC delta was +0.004784, but its grouped interval [-0.004081, +0.015125] crossed zero. Nuisance/competition had the strongest family point estimate, also with an interval crossing zero. Scientific promotion and the full model-annotation audit remain incomplete.

[Result and provenance](../../docs/research/SPON_CA_BURST_FEATURE_ATLAS_V1_RESULTS.md) · [Decision](../registry/decisions/NREV-DEC-0028.yaml)

**Boundary:** SPU-AUC measures positive-versus-unlabeled ranking, not precision, specificity, calibrated neuron probability, or end-to-end proposal recall. The grouped global-AUC interval and macro held-fold AUC are distinct summaries and must retain their names. The final Run E changes a display bound only; failed partial runs and superseded visual iterations remain preserved. The new ranking audit is incomplete and the 78 positive anchors remain part of the same historically examined recording.


## Current decision queue

1. **Complete External Bounded Review V2:** Obtain two locked Phase A and Phase B submissions, run agreement analysis, adjudicate all spatial, class, and identity disagreements, and freeze a local truth revision before any detector or learning evaluation.
2. **Uncertainty-Aware Positive-Unlabeled Feature Fusion V1 with v1.1 output-assembly amendment:** Run the prespecified harmonized feature audit, positive-unlabeled learning, label-sensitivity models, leakage-resistant grouped validation, feature stability, counterfactual perturbation, and hard-subgroup tests without promoting provisional labels.
3. **Identity Aware Bounded Review:** Exhaustively review the ambiguous ROI 10/15, ROI 11, ROI 12, and ROI 23 clusters with joint footprint and identity assignment.
4. **Bounded Field Precision:** Exhaustively review a bounded field and estimate precision with verified negatives.
5. **Independent Recording Confirmation:** Apply frozen pipeline and phenotype assignments to an independent recording.
6. **Motion and registration-confound audit v2:** Export motion fields and registration residuals and relate them to shared waveform strength.
7. **Frozen Feature Transfer:** Transfer the frozen carrier, coherence, recurrence, consensus, and persistence definitions without tuning and test task-specific utility.
8. **Realistic Movie Simulation:** Preserve the current spatial-context and robust-stopping baseline; defer additional deblender complexity until bounded biological identity truth or a genuinely independent labelled recording is available.
- **Compact Spatiotemporal JEPA Representation Pilot V1:** Test a bounded raw-only JEPA representation against capacity-matched MAE and random encoders under one common frozen primary head and against a pair-safe source-off-calibrated handcrafted comparator, while keeping objective-native error heads secondary. _(registered draft; scientific priority not assigned)_
- **Frozen-JEPA Conditional-Background Residual Screen V1 with v1.1 domain amendment:** Run a bounded one-seed engineering screen of normalized signed video-minus-conditional-prediction residuals with exact target isolation, source-off-only calibration, paired signal-retention accounting, and no scientific promotion. _(registered draft; scientific priority not assigned)_
- **Source-off conditional-background predictor feasibility v1:** Run a bounded label-free source-off feasibility benchmark before authorizing any new conditional-background residual detector experiment. _(registered draft; scientific priority not assigned)_
- **Contextual-envelope feature semantics pilot v1:** Separate instantaneous evidence, contextual upper envelope, envelope agreement, envelope-gated evidence, and agreement-attenuated evidence across frozen Raw and learned representations. _(registered draft; scientific priority not assigned)_
- **Contextual-envelope retrieval comparison v1:** Evaluate temporal-only and spatiotemporal agreement attenuation with amplitude, shuffled-envelope, and displaced-envelope controls over all canonical-v7 occurrences. _(registered draft; scientific priority not assigned)_

See [the claim ledger](CLAIM_LEDGER.md) and [experiment timeline](EXPERIMENT_TIMELINE.md) for the complete generated record.
