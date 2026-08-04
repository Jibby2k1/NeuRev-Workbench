# Asset manifest and placeholders

All entries below are placeholders. Generated sources are under
`Outputs/HierarchicalParzenICA/spon_ca_burst_joint_msln_residual_sweep_v2/` and
are ignored by Git. Copy selected assets into `assets/` manually.

## Required videos

| Placeholder target | Generated source | Use |
|---|---|---|
| `assets/VIDEO_01_finalist_comparison.mp4` | `videos/finalist_persistence_innovation_comparison.mp4` | Primary paper supplement and slide 5 |
| `assets/VIDEO_02_s15_g3_t31_layer_journey.mp4` | `videos/joint_s15_g3_t31_g1_layer_journey.mp4` | Leading broad-context layer journey |
| `assets/VIDEO_03_s15_g3_t23_layer_journey.mp4` | `videos/joint_s15_g3_t23_g1_layer_journey.mp4` | Broad-context sensitivity comparison |
| `assets/VIDEO_04_s5_g1_t15_layer_journey.mp4` | `videos/joint_s5_g1_t15_g1_layer_journey.mp4` | Compact-context comparison |

All four sources are H.264, 960x684, 560 frames, 10 fps, and 56 seconds. They
cover UI frames 1800–2359; playback is five times slower than the 50 Hz
acquisition.

**Video caption:** “Synchronized review of causal joint normalization and
CS-Parzen persistence/innovation across the four annotated burst intervals.
Display ranges are fixed for the full interval. Rings denote sparse known
positives only; unmarked pixels are unknown.”

## Required figures

| Placeholder target | How to make it | Use |
|---|---|---|
| `assets/FIGURE_01_method_schematic.png` | Draw from the causal exclusions and pipeline in the frozen v2 manifest/workflow | Paper Fig. 1; slides 2–3 |
| `assets/FIGURE_02_context_screen.png` | Compose `diagnostics/stage_a/joint_bank_ui_{1900,2003,2040,2122,2254}.png` | Paper Fig. 2; slide 4 |
| `assets/FIGURE_03_representative_bursts.png` | Export matched frames from `VIDEO_01` at UI frames 2003, 2040, 2122, and 2254 | Paper Fig. 3; slide 5 |
| `assets/FIGURE_04_budget_guardrails.png` | Plot `stage_c/ica_metrics.json` matches by budget for finalist lanes | Paper Fig. 4; slide 6 |
| `assets/FIGURE_05_stability_and_cuda.png` | Plot bootstrap angles plus values from `gpu_validation.json` | Paper Fig. 5; slide 7 |

**Figure 1 caption:** “Causal joint spatiotemporal normalization excludes the
current frame, a protected spatial core, and the most recent temporal guard
frame. The signed residual feeds a bounded gate and two separately retained ICA
coordinates interpreted operationally as persistence and innovation.”

**Figure 2 caption:** “Bounded screen of 30 causal joint contexts. Visual and
morphology proxies selected diverse contexts for gating and three finalists for
ICA confirmation; labels were not used for fitting.”

**Figure 3 caption:** “Representative burst frames show that broad persistence
lanes retain coherent anatomical activity while innovation emphasizes different
temporal structure. Visual interpretation is primary because sparse annotations
do not define negatives.”

**Figure 4 caption:** “Sparse-positive known-label matches across fixed per-burst
candidate budgets. The Raw Direct result of 49/79 at 232 quiet-threshold
proposals is an external anchor, not a protocol-identical comparator.”

**Figure 5 caption:** “CUDA acceleration preserved CPU numerical references
within measured tolerances and reduced full-context normalization time from
5.29 s to 0.44 s. Bootstrap-angle dispersion limits claims of stable broad ICA
source recovery.”

## Optional source records

For reproducibility, archive copies of `gpu_validation.json`, `run_manifest.json`,
`stage_a/metrics.json`, `stage_b/gate_metrics.json`, `stage_c/ica_metrics.json`,
and `videos/video_manifest.json` alongside the final submission. Do not replace
the repository’s ignored completed output root or edit its contents in place.
