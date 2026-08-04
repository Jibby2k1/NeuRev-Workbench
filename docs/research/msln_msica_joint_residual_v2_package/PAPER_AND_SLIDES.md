# Paper and slide plan

## Paper: minimal figure sequence

1. **Method.** Show Raw, the excluded current/core/guard samples, causal joint
   `Zst`, bounded gate, and the separate persistence/innovation outputs.
2. **Context screen.** Compare the 30 contexts and identify why broad
   `S15/G3/T31`, broad `S15/G3/T23`, and compact `S5/G1/T15` were retained.
3. **Representative bursts.** Use identical frames and fixed display ranges to
   compare Raw, `Zst`, gated Raw, persistence, and innovation.
4. **Guardrail curves.** Plot known-label matches versus candidate budget for
   the three finalists, with Raw Direct shown as a non-identical external anchor.
5. **Stability and compute.** Pair bootstrap-angle distributions with the CUDA
   parity, speed, and peak-allocation summary.

Suggested title: **Causal Joint Spatiotemporal Normalization with Persistence–Innovation Separation for Calcium Activity Review**.

Suggested results sentence: “At a fixed allocation of 58 candidates per burst
(232 total), the broad `S15/G3/T31` persistence lane matched 58/79 sparse known
positives, compared with the external Raw Direct anchor of 49/79 from 232
quiet-threshold proposals; because proposal allocation differed and negatives
were unlabeled, this comparison was treated as a recall guardrail.”

## PowerPoint: eight-slide version

1. Problem: coherent activity is mixed with background and temporal nuisance.
2. Design principle: causal reference with protected spatial and temporal guards.
3. Pipeline: signed `Zst` to gate to separate persistence and innovation.
4. Context search: 30 bounded contexts and three finalists.
5. Visual result: finalist comparison video, paused at representative bursts.
6. Quantitative guardrail: 58/79, 56/79, and 47/79 at budget 58.
7. Engineering result: 12.0x normalization speedup, CPU/GPU parity, 6.35 GiB
   observed peak under an 8 GiB cap, and resumable sequential Stage C.
8. Conclusion: strong visual candidate; exhaustive review and stability
   confirmation remain before a biological or precision claim.

## Statements to avoid

- “ICA recovered the true neural source.”
- “The method has higher precision than Raw Direct.”
- “58/79 versus 49/79 is a controlled head-to-head win.”
- “Innovation failed.” It had low sparse-label recall but can still visualize
  onset-like structure.
- “GPU and CPU are identical.” State the measured numerical tolerances instead.
