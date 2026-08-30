# Spon Ca Burst major validation next steps v1

## Outcome

All three priority tracks were advanced, but they ended at different evidence gates.

1. **Independent transfer:** 11 untouched 060126 TIFF recordings and two ROI workbooks were inventoried and hashed. Transfer was deliberately not executed because these are rest/left/right behavioral recordings and no compatible, identity-resolved spontaneous-burst event truth was found. This is a blocked preflight, not a failed replication.
2. **Identity-complete bounded truth:** a detector-blind two-reviewer packet, coverage signoff, explicit identity/overlap classes, and adjudication template are ready. Precision remains unavailable until both reviewers complete the full region and conflicts are adjudicated.
3. **Realistic movie validation:** 432 paired conditions were completed (12 seeds, 36 factorial cells per seed). Conditions varied source distance (3, 6, 10 px), weak/strong amplitude ratio (0.35, 0.65, 1.0), temporal correlation (0, 0.7), and weak-source footprint (ellipse or crescent), with structured background, motion, signal-dependent noise, and read noise.

## Simulator result

Mean source-pixel AUC was 0.813 for pixel variance and 0.917 for spatial context. The paired mean gain was +0.104, the empirical central 95% interval was -0.018 to +0.210, and spatial context won 94.9% of paired cells.

This supports the current feature story: spatial context is not merely decorative morphology; it can materially improve discrimination under controlled crowding, shape, motion, and noise. The lower interval crossing zero also identifies regimes where context is neutral or harmful, so it should remain a role-specific feature rather than replace carrier evidence wholesale.

## Evidence boundary and next gate

The simulator supplies exact source identity but is not biological external validation. The bounded packet can establish local precision only after human completion. A genuine frozen transfer requires an independent recording with a declared event-window contract and identity-resolved truth; the existing behavioral files cannot be substituted without changing the scientific task.

The next simulator increment should evaluate complete proposals rather than pixels: one-to-one identity assignment, localization error, recall and precision at fixed budgets, duplicate suppression, weak-neighbor recovery, and calibration/abstention. That is the appropriate place to compare carrier-only, spatial-context, kinetic, and joint feature stacks under a fully frozen detector.
