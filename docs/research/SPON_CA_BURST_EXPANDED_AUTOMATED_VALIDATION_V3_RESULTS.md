# Expanded automated validation v3

This frozen secondary suite evaluates ten non-reinforcement-learning test
families on the canonical-v8 evidence base. It uses 106 occurrences at 50
identity-grouped sites, preserves the 94 recovered and 12 missed B58 outcomes,
and reuses the validated visual audit. No detector or label was refit.

## Main results

- Five-fold identity-grouped prediction reached ROC AUC 0.833 and log loss
  0.498 with the full feature panel, versus AUC 0.611 and log loss 0.690 with
  carrier alone.
- Across 100 site-bootstrap sparse fits, variance-stabilized center-annulus was
  selected in 97%, multiscale persistence in 73%, lag-2 recurrence in 39%, and
  the matched filter in 29%. The annular feature is interpreted as persistent
  observability or acquisition context, not automatically neuronal signal.
- Within-burst conditional permutation increased log loss most for
  variance-stabilized annular context (0.0484), persistence (0.0244), lag
  recurrence (0.0234), LS center (0.0164), and coherence (0.0164).
- Confidence-based abstention reduced error from 20.8% at full coverage to
  9.4% at 50% coverage. This is an internal selective-risk result, not
  calibrated deployment performance.
- Recovered-only anomaly detection separated misses with AUC 0.814 for
  isolation forest and 0.760 for robust distance, supporting label-light review
  prioritization.
- The simple five-neighbor spatial graph statistic had essentially no linear
  association with recovery (-0.009); distance-only crowding is insufficient.
- Controlled one-dimensional mixtures showed that increasing a neighboring
  transient can raise peak/MAD even without improving target identity. Frozen
  intensity scaling preserved peak/MAD, added common mode reduced it and raised
  lag recurrence, and temporal shifts changed fixed-window retrieval as
  expected.
- Cross-burst transfer remains structurally underpowered for identity-clear
  misses: only one clear source-target pair exists. Pre-event-frozen footprints
  remain the usable prospective within-recording alternative.

## Major decisions

1. Transfer the frozen measurement stack to an independently acquired
   recording, retaining identity/site grouping and abstention evaluation.
2. Create an exhaustively reviewed bounded field with explicit one-to-one,
   overlapping, duplicate, uncertain, and non-neuronal identity states.
3. Build an anatomy-, optics-, motion-, kinetics-, and noise-aware movie
   simulator with exact source identity, then compare compact operators,
   source-separation methods, and identity-aware selection under controlled
   overlap.

Further same-recording feature expansion is secondary to these three evidence
gaps.
