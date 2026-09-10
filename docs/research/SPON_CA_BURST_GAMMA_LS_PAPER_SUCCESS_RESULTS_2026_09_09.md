# Spon Ca Burst Gamma-LS paper-success campaign: final results

**Finalized:** 2026-09-09

**Frozen design:** [campaign contract](SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_CAMPAIGN_2026_09_08.md)

**Repository snapshot:** branch `codex/neuron-identifiability-paper-20260822`, commit `0fe1b38aec6e1748fa5ec056aa883977d9999f05`; the working tree is intentionally dirty, so the hash-bound artifact indexes below—not the Git commit alone—are the result authority.

## Technical summary

The smallest defensible paper architecture is a **single-scale, causal signed-difference pipeline with radial Gamma-weighted local standardization (Gamma-LS)**. Two-frame CS-Parzen ICA did not provide a protected, identity-paired gain over signed differencing, and PCA was used only as a whitening/control construction; neither PCA nor ICA belongs in the deployed path. Max pooling was excluded by the frozen contract.

The common operational Gamma-LS context is half-width (h=15) (31-pixel support), 7-pixel guard, Gamma shape (n=9), and mode radius 7.5 px. It was the coordinate-free training winner in all four folds, but it **did not pass the predeclared support-sufficiency rule**. The largest tested boundary, (h=31) (63-pixel support), was clearly inferior in all four training folds, so the contract did not trigger testing at (h=39) or (h=47). Thus there is no evidence that a support larger than (h=31) is needed, but there is also no basis for calling (h=15) sufficient or optimal.

At the operational (q=1) calibration state, the automated full-record ledger contains **371 frame-level proposals over 2,259 eligible application frames** (0.1642 proposals/frame; 295 frames with at least one proposal). The final model-only scientific audit passed with 24 label-free candidate surrogates from 62 spatial clusters, 25 fully decoded videos, 27 PNGs, no annotation source opened, no temporal linking, and completed manual visual QA. These are not 371 unique neurons, biological events, or confirmed detections. On the separate `15 right` recording, the frozen B58-per-block readout matched 6 of 51 sparse-positive occurrences (recall 0.1176; mean reciprocal block rank 0.0356). Unmatched proposals remain unknown, so precision, specificity, and false-positive rate are not identified.

On an RTX 4070 SUPER with PyTorch 2.11.0+cu130, corrected matched-workload batch throughput was 1,470--2,216 frames/s across batches 1--64. However, two 60-s paced trials both failed the predeclared strict 1-kHz gate: one had 212/60,000 deadline failures, and the second had 9,327/60,000 failures including 37 drops. The paper may claim fast GPU execution and batch throughput, but not sustained 1-kHz readiness.

## Canonical architecture and readout definitions

```text
acquired uint16 frame
  -> Gaussian smoothing (sigma=1 px, reflect boundary, truncate=4; 9x9 support)
  -> causal EMA (alpha=0.4, exact state carry)
  -> signed adjacent difference
  -> guarded radial Gamma-LS (h=15, guard=7, n=9, mode=7.5,
                              valid-reference renormalization, epsilon=1e-6)
  -> empirical one-sided threshold
  -> deterministic NMS (6-px image-border exclusion and 6-px Euclidean separation)
  -> automated frame-level proposals
```

This is **single-scale inference**. The campaign evaluated multiple support sizes and selected one fixed context; it did not fuse or jointly apply features at multiple scales.

Three fixed evaluation heads must remain separate:

| Scope | Fixed readout | Meaning of (q) or budget |
|---|---|---|
| Protected four-burst evaluation | Threshold each aligned frame, average binary exceedance into one burst-occupancy map, then apply one NMS per burst | Target NMS peaks per duration-matched quiet pseudo-burst; B20/40/58/80/100 are peaks retained per burst |
| Independent `15 right` evaluation | Pool each one-second block with LME exponent 0.25, then NMS and retain B candidates per block | Candidate budget per complete one-second block |
| Operational full-record ledger | Threshold and NMS every eligible frame | Target peaks per initialization calibration block; output rows are frame-local proposals |

The empirical burden parameter is a calibration target, not a probability-of-false-alarm guarantee. Application burden can differ from the calibration burden.

## Exact campaign status

Here, **artifact PASS** means that the saved files, counts, hashes, boundaries, and declared contract reconcile. It does not convert a failed scientific gate into a positive result.

| Evidence block | Artifact status | Scientific decision | Authority |
|---|---|---|---|
| Operator and CUDA regression | **PASS:** 231 tests, 0 failures/errors/skips, 4.599 s, on the host-visible CUDA runtime | CPU/CUDA, signed-input, boundary, chunking, NMS, streaming, protected-join, and artifact invariants covered by the focused Gamma-LS suite; this is not the full repository suite | JUnit XML (`Outputs/GammaLSDifference/gamma_ls_final_cuda_regression_20260908.xml`) |
| Coordinate-free support grid | **PASS:** (h=11,15,19,23,31); controls executed; no protected coordinates/identities used | (h=15) won all four training folds; (h=31) was not best or near-optimal in any fold, so no larger extension was triggered | support screen (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_support_sufficiency_v1_gpu_screen_20260908_r2/REPORT.md`) |
| Protected support rule | **PASS artifact; FAIL decision** | `unresolved_failed_protected_sensitivity_and_repeated_latency`; (h=15) is operational, not proven sufficient | protected support analysis (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_protected_support_sufficiency_analysis_v1_20260908_r1/REPORT.md`) |
| Two-frame representation ablation | **PASS metric artifact** | CS-Parzen ICA did not beat signed differencing under the protected paired rule; PCA derivative was nearly a scaled signed difference | summary (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_protected_20260908_r2/summary.json`), validation (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_protected_20260908_r2/validation.json`) |
| Matched six-lag ablation | **PASS metric artifact** | Low-burden CS-Parzen residual gains over its PCA control were promising but nonuniform; the full promotion gate failed, so this remains supplementary/exploratory | report (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_multilag_protected_20260908_r1/REPORT.md`) |
| Conditioning grid | **PASS screen and protected metric artifacts** | No candidate improved all five protected burdens for any representation; retain sigma 1 / EMA 0.4 | protected report (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_protected_v1_gpu_20260908_r1/REPORT.md`) |
| Simple operator controls | **PASS metric artifact** | Radial Gamma-LS superiority was not established; square-annulus was broadly similar, while positive box-CFAR was faster but weaker on differenced inputs | control report (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_protected_simple_controls_v1_20260908_r1/REPORT.md`) |
| Exact-truth simulator | **PASS metric artifact** | Radial was the descriptive aggregate leader at the current-frame (q=1) endpoint but did not dominate every nuisance family; synthetic truth does not validate biological precision | exact-truth report (`Outputs/GammaLSDifference/gamma_ls_exact_truth_framewise_v1_results_20260908/REPORT.md`) |
| Fixed all-four-burst replay | **PASS metric and audit-input artifact** | Post-selection, within-recording characterization only; not a protected selection estimate | fixed replay (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_fixed_deployment_characterization_v1_20260908_r1/REPORT.md`) |
| Fixed-replay scientific audit | **PASS:** inventory, decode, annotation separation, exact-stage provenance, and manual visual QA | Supports the protected burst-occupancy figure/readout; does not cover the operational full-record proposal stream | audit report (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_fixed_deployment_scientific_audit_v1_20260908_r2/REPORT.md`) |
| Full-record automated ledger | **PASS metric artifact** | 371 frame-level proposals on 295 of 2,259 eligible frames at operational (q=1); proposal count only | ledger (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_full_recording_signed_h15_v1_20260908_r2/REPORT.md`) |
| Full-record model-only scientific audit | **PASS:** exact-stage replay, inventory, 25/25 full video decodes, 27 PNGs, annotation separation, and manual visual QA | Audits 371 frame-level proposals on 295/2,259 frames through 24 label-free surrogates from 62 spatial clusters; no temporal linking and no biological-detection claim | final audit (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_full_recording_q1_model_only_scientific_audit_v1_20260909_r1/REPORT.md`) |
| Independent `15 right` metrics | **PASS metric artifact** | 6/51 at B58; one-recording sparse-positive sensitivity only | independent report (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_independent_15_right_v1_20260908_r1/REPORT.md`) |
| Independent scientific audit | **PASS:** inventory, decode, marker separation, numerical replay, and manual visual QA | Makes the one-recording result auditable; does not establish population generalization | audit report (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_independent_15_right_scientific_audit_v1_20260908_r2/REPORT.md`) |
| Exact deployed timing | **PASS artifacts; FAIL 1-kHz decision in both trials** | Corrected batch frontier is valid for throughput; paced service is not zero-miss/zero-drop 1-kHz ready | r1 validation (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_exact_nms_streaming_signed_h15_1khz_20260908_r1/validation.json`), r2 validation (`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_exact_nms_streaming_signed_h15_1khz_20260908_r2/validation.json`) |
| Max pooling | Not run by design | Excluded from the compact paper because the frozen campaign did not justify another pooling branch | [campaign contract](SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_CAMPAIGN_2026_09_08.md) |

## Representation decision: deploy signed differencing, not ICA

The protected primary population contains 79 occurrences from 26 canonical identities. At NMS 6 and B58, the cross-fit-average **pooled known-positive recall** was:

| Quiet burden (q) | Signed difference | PCA-whitened derivative | Two-frame CS-Parzen ICA |
|---:|---:|---:|---:|
| 0.25 | 0.0506 | 0.0506 | 0.0696 |
| 0.5 | 0.0506 | 0.0506 | 0.0823 |
| 1 | 0.0886 | 0.0949 | 0.0949 |
| 2 | 0.1392 | 0.1392 | 0.1456 |
| 5 | 0.1835 | 0.1835 | 0.1962 |

All candidate sets contained fewer than 20 retained peaks, so B20--B100 were saturated. B58 is a consistent display point, not independent evidence beyond the burden curve.

The identity-clustered, cross-fit-average ICA-minus-signed macro-recall contrasts were 0.0208 [-0.0163, 0.0861], 0.0327 [-0.0078, 0.0983], 0.0071 [-0.0150, 0.0385], 0.0042 [-0.0202, 0.0382], and 0.0086 [-0.0188, 0.0441] for (q=0.25,0.5,1,2,5), respectively. Every interval crossed zero. The learned two-frame coordinate also varied strongly by fold: its Pearson correlation with signed difference was approximately 0.110, 0, 0.229, and 0.999. It is therefore incorrect to describe the learned CS-Parzen component as generally rediscovering the derivative.

PCA has a narrower interpretation. Full-rank PCA whitening was introduced only to construct a matched linear control before applying the derivative direction; it was not an unseen deployed preprocessing stage. Within the fold-local selected support role, the PCA derivative correlated 0.999629--0.999996 with signed difference and produced nearly identical protected recall. This confirms that the two-frame PCA control is effectively a rescaled derivative in this experiment, not that PCA adds a useful deployed feature.

The separate six-lag experiment is not a reason to reinsert ICA into the main architecture. CS-Parzen delay residual exceeded its PCA delay-energy control at (q=0.25) by 0.05285 [0.00658, 0.11432] and at (q=0.5) by 0.04823 [0.00568, 0.09973]. The (q=1) interval touched zero, and point effects at (q=2) and (q=5) were slightly negative with intervals spanning zero. This is a bounded low-burden signal that merits future confirmation, not a passed all-burden promotion gate.

## Support-size decision: h15 is operational, not sufficient

The coordinate-free screen selected a 31-pixel support ((h=15)), guard 7, and shape 9 in all four training folds. Mode fraction 0.5 won three folds; mode fraction 0.75 won one. The common deployment freezes the three-fold context, `support_support_a_h15_g7_n9_m0p5`, rather than claiming fold-specific multiscale inference.

The (h=31) endpoint was outside the near-optimal tolerance in all four folds, with training contrasts -0.0280, -0.00320, -0.0498, and -0.0361. The stopping rule therefore closed the upper boundary without testing (h=39) or (h=47). This supports only the statement that the screen found no evidence requiring supports beyond the tested (h=31) boundary.

Formal sufficiency nevertheless failed after the frozen candidates were joined to protected labels. The smaller candidate was detectably worse than its larger comparator for the energy-normalized-difference arm at (q=0.25) and 0.5 (delta -0.01786, 95% CI [-0.03750, -0.000740]) and at (q=2) (delta -0.02976, 95% CI [-0.05952, -0.00543]). The analysis also recorded 10 swap-specific practical-tolerance failures.

Repeated one-frame Gamma-stage timing was non-monotone and did not resolve the tradeoff:

| Half-width | Support width | p50 (ms) | p99 (ms) |
|---:|---:|---:|---:|
| 11 | 23 | 1.8045 | 2.4539 |
| 15 | 31 | 0.8681 | 1.2913 |
| 19 | 39 | 1.2194 | 1.6788 |
| 23 | 47 | 1.6819 | 2.3209 |
| 31 | 63 | 2.8286 | 3.4610 |

These numbers cover the Gamma stage only. No compared radial width passed the predeclared p99 <= 1 ms gate in every required comparison. The paper should present (h=15) as a **fixed operational context chosen from the coordinate-free training screen**, not as the smallest sufficient or globally optimal reference.

## Controls and exact-truth boundary

The signed square-annulus control broadly matched the radial operator on sparse-positive recall, and was sometimes stronger at individual points. Because the parent campaign did not predeclare a noninferiority margin, this remains descriptive and does not establish formal equivalence. The maintained positive-clipped box-CFAR control was much faster in the isolated repeated test (p50 0.2138 ms versus 1.6579 ms for square-annulus and 1.8045 ms for radial (h=11)), but its protected point recall was lower on both differenced representations; uncertainty was wide for signed difference and clearer for the energy-normalized arm. The radial operator is retained as the frozen, interpretable implementation—not as a universally superior detector.

The exact-truth benchmark used 108 generated movies, six nuisance families, 95 scored frames/movie, and exhaustive active-frame source truth. At one calibrated null proposal per frame, the current-frame radial arm had aggregate F1 0.78186, compared with 0.77471 for positive box-CFAR and 0.77314 for square-annulus. Radial did not dominate every family; dense neuropil favored the box control. Signed-difference arms are disadvantaged by this active-frame endpoint because they emphasize changes/onsets. These synthetic precision and localization results apply only to the simulator and cannot be transferred to biological precision.

## Automated output and independent-recording result

For the complete 2,359-frame Spon Ca Burst acquisition, causal state was initialized from UI frame 1. The operational variant used UI frames 1--100 only for annotation-file-and-location-free empirical calibration and applied the frozen detector to UI frames 101--2359:

| Calibration target (q) | Eligible frames | Frame-level proposals | Proposals/frame |
|---:|---:|---:|---:|
| 0.25 | 2,259 | 268 | 0.1186 |
| 0.5 | 2,259 | 271 | 0.1200 |
| 1 | 2,259 | 371 | 0.1642 |
| 2 | 2,259 | 401 | 0.1775 |
| 5 | 2,259 | 970 | 0.4294 |

At (q=1), those 371 proposal rows occur on 295 frames. The first 100 frames were not assumed biologically quiet; (q) is an empirical initialization-block calibration target. No temporal linking was run. The final model-only audit at `Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_full_recording_q1_model_only_scientific_audit_v1_20260909_r1` passed exact-stage and ledger reconciliation, retained 24 label-free candidate surrogates from 62 spatial clusters, fully decoded all 25 videos, indexed 27 PNGs, opened no annotation source, and passed manual visual QA. Its artifact-index SHA-256 is `bfa60a46d13ba2e7f1a9ca6d3642a1e59c27b96513a296cacb9333ca2bd4643e`; its validation SHA-256 is `4d1d633491b92d167c84e1e8a81ebdd7697e965ff5e7d9cffe7d9f7f98aa20f6`. This closes the full-record model-only media gate, but it audits frame-level proposals rather than establishing unique biological detections, precision, specificity, false-positive rate, or exhaustive recovery under sparse labels.

On `15 right`, the frozen signed-difference/Gamma-LS arm produced 6,400 candidates over 32 complete one-second blocks. At the frozen B58 readout, 1,856 candidates were retained and 6/51 sparse-positive occurrences matched one-to-one (recall 0.117647; mean reciprocal block rank 0.035551). At B200, 11/51 matched. This is one-recording sensitivity evidence: calibration and evaluation come from the same recording, and the eligibility preflight inspected annotation content before the subsequent label-isolated scoring phase. The independent scientific audit passed with 110 PNGs, 36 fully decoded MP4s, separated expert/model markers, manual visual inspection, and a CPU FFT replay maximum absolute error of 6.44e-6 against the production score (tolerance 1e-4).

## Fixed deployment replay and scientific audits

The fixed all-four-burst replay sealed 138 candidates, 10 calibration states, 40 occupancy maps, 3,950 occurrence-match rows, and 25 cross-fit curve rows before parsing 79 protected occurrences. At (q=1)/B58, `a_train_b_test` matched 6/79 with 9 effective candidates, `b_train_a_test` matched 7/79 with 10, and the paired quiet-swap mean recall was 0.08228. This is explicitly post-selection within-recording characterization.

The metric artifact passed all 14 validation checks and is bound by artifact-index SHA-256 `42c90515c1d72630ff62abed58f2f3c8f14c2f3ce5fb7831030317f68ce2fff9`. The separate r2 scientific audit passed inventory/decode and visual QA with 120 PNGs and 33 MP4s. Its same-crop stage packet uses exact acquired raw data and hash-sealed conditioned, signed-difference, Gamma-LS, and protected burst-occupancy arrays. A derived same-frame threshold/NMS diagnostic is labeled separately from both the protected burst-occupancy head and the operational full-record head.

## GPU efficiency: high batch throughput, failed strict streaming gate

Both exact timing trials used an NVIDIA GeForce RTX 4070 SUPER, driver 580.173.02, PyTorch 2.11.0+cu130, and CUDA 13.0. The service boundary includes pinned-host transfer, causal GPU preprocessing, the fixed radial Gamma-LS operator, strict thresholding, GPU local-maximum extraction, transfer of the complete sparse candidate set, and deterministic CPU Euclidean cleanup. It excludes camera/disk acquisition, artifact serialization, and downstream control.

| 60-s paced trial | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Processed / scheduled | Missed processed deadlines | Drops | Total failures | Max queue / backlog | Strict 1-kHz gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1 | 0.6752 | 0.7039 | 0.7547 | 1.7484 | 60,000 / 60,000 | 212 | 0 | 212 (0.353%) | 2 / 1 | **FAIL** |
| r2 | 0.6807 | 1.1043 | 1.7657 | 4.2705 | 59,963 / 60,000 | 9,290 | 37 | 9,327 (15.545%) | 8 / 7 | **FAIL** |

The stable median and divergent tail show why a single fast trial or mean throughput cannot support a real-time readiness claim. Both valid paced outcomes must be reported.

The r1 batch table is not suitable for cross-batch comparison because batch 1 repeatedly used one frame, yielding a zero recurrent difference and zero candidates. The corrected r2 frontier gives every batch size the same 128,000 sequential cyclic real frames and the same 672,000 retained candidates:

| Batch | Frames/s | Amortized ms/frame |
|---:|---:|---:|
| 1 | 1,470.317 | 0.6801 |
| 8 | 2,215.762 | 0.4513 |
| 32 | 2,141.722 | 0.4669 |
| 64 | 2,062.989 | 0.4847 |

This establishes GPU batch throughput above 2,000 frames/s for batches 8--64 on the measured boundary. It does not establish zero-miss one-frame arrivals, acquisition-to-control latency, or voltage-imaging inverse-control readiness.

## Paper-safe claims and prohibited upgrades

| Supported wording | Do not upgrade to |
|---|---|
| “Causal neural event **candidate extraction** with Gamma-weighted local standardization” | “Quadratic Gamma Detector” as the current system name |
| “A single fixed Gamma-LS scale was selected after evaluating multiple contexts” | “Multiscale inference” or “multiscale features” |
| “Signed temporal differencing was retained after protected ICA/PCA ablations” | “ICA is part of the deployed architecture” or “ICA improves detection” |
| “PCA whitening was a matched ablation/control and produced a nearly scaled derivative” | “PCA is a hidden deployed preprocessing stage” |
| “371 automated frame-level proposals over 2,259 eligible frames” | “371 detections,” “371 neurons,” or “371 biological events” |
| “Known-positive recall under sparse annotation; unmatched proposals are unknown” | Precision, specificity, false-positive rate, or exhaustive recovery |
| “One-recording `15 right` sparse-positive sensitivity” | Cross-animal or population generalization |
| “1,470--2,216 frames/s corrected batch throughput; median paced latency about 0.68 ms” | “1-kHz ready,” “real-time guaranteed,” or “closed-loop ready” |
| “(h=15) is the frozen operational context; no screen evidence required (h>31)” | “(h=15) is sufficient,” “minimal,” or “optimal” |
| “Radial Gamma-LS is an interpretable frozen implementation” | “Radial Gamma weighting is superior to square or box controls” |

## Remaining claim gates and next experiments

The compact architecture/applied paper can proceed now if it uses the wording above and treats negative results as design simplification. The full-record model-only scientific-media gate is closed by the final audit. The following gates remain before stronger claims:

1. **Sustained 1-kHz readiness:** optimize and repeat one-frame service under a predeclared jitter-controlled protocol until p99 is below 1 ms with zero deadline misses, zero drops, and no backlog in repeated trials. A later end-to-end test must add acquisition and downstream control.
2. **Precision and generalization:** exhaustively adjudicate a frozen proposal sample to identify precision/FPR, and repeat the frozen pipeline on additional recordings/animals with untouched selection. Sparse-positive recall alone cannot close either gate.
3. **Support sufficiency:** if the manuscript needs a sufficiency/minimality claim, predeclare a new paired comparison with raw latency samples and a protected noninferiority rule. The present grid supports operational use of (h=15), not sufficiency.
4. **ICA promotion:** only revisit six-lag ICA if it receives an independent, all-burden confirmation plus scientific audit and a demonstrated latency benefit/acceptable cost. Two-frame ICA should not be promoted.
5. **Operator superiority:** a radial-versus-square/box superiority or noninferiority claim requires a predeclared margin and independent confirmation. It is not necessary for the current architecture paper.
6. **Metric-only ablation audits:** the representation, conditioning, simple-control, multilag, and exact-truth artifacts pass their metric contracts but do not yet have complete scientific-media audits. They may support conservative exclusion/supplementary statements, but any headline promotion would require the corresponding audit to be completed.

## Reproducibility anchors

- Final focused CUDA regression XML: SHA-256 `62bd607a4304334ce18017c85eba2ece9ccc09479f00a6e96129715292665729` (35,006 bytes).
- Primary Spon Ca Burst source movie: SHA-256 `04dfbe2f7cb69d72ff75e23ad17c87b3fd406c96a4b872148de2285a9a44d449`.
- Support-screen artifact index: SHA-256 `ae0337597ccd7b21a7e92f910f568809a57895b68df752f73940b7b830db7e83`.
- Protected representation artifact index: SHA-256 `ab4816cb0d1548475c55dd712409a9dd8842caf1174cbf50d93096b470d1560e`.
- Full-record proposal-ledger artifact index: SHA-256 `cc1cdfa70732204cf2c8e73e72f90b8cf5f06acf2193fa56218be6c4e9a6bced`.
- Full-record model-only scientific-audit root: `Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_full_recording_q1_model_only_scientific_audit_v1_20260909_r1`; artifact-index SHA-256 `bfa60a46d13ba2e7f1a9ca6d3642a1e59c27b96513a296cacb9333ca2bd4643e`; validation SHA-256 `4d1d633491b92d167c84e1e8a81ebdd7697e965ff5e7d9cffe7d9f7f98aa20f6`.
- Corrected exact-timing r2 artifact index: SHA-256 `0795190555b8c3205cf0e1939c7b1fb86c29f9b686a232347642343d1bb8b1f3`.
- Fixed-deployment replay artifact index: SHA-256 `42c90515c1d72630ff62abed58f2f3c8f14c2f3ce5fb7831030317f68ce2fff9`.
