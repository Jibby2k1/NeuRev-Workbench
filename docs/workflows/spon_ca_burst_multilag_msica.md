# Spon Ca Burst multi-lag MSICA v5

## Purpose

This workflow compares higher-order temporal MSICA representations in two
architectures:

1. Raw -> MSICA.
2. Raw -> MSICA -> MSLN.

The adjacent two-frame CS-Parzen model is a historical control, not the new
experimental treatment. The v5 treatments are a shared two-output demixer
optimized over several temporal lags and a full delay embedding separated into
persistence, innovation, and residual-subspace energy.

## Frozen design

The manifest is
examples/spon_ca_burst_multilag_msica_v5.example.json. It evaluates CS-Parzen,
KSG mutual information, normalized HSIC, and matrix-Renyi mutual information.
Lag profiles reach 16 frames (320 ms); the dense embedding also tests
0,1,2,3,4,6,8 frames. Parameter calibration is followed by expansion over lag
profiles and lag weighting.

Labels are sealed through parameter calibration, representation scoring,
five-seed real-data confirmation, five-seed synthetic recovery, and all 30
MSLN contexts. Each lane stores its top 100 spatial proposals per burst. Only
after every choice is frozen are those proposals matched to the 79 sparse
known-positive labels. Unmatched candidates remain unknown.

## Resource contract

Use .venv-neurobench/bin/python. The full run requires explicit Spon
authorization, one CUDA worker, at most four numerical-library threads, an
8 GiB experiment VRAM cap, and eight-frame projection chunks. Run preflight and
gpu-preflight before surface, confirm, pipeline, and finalize. Never reuse an
existing output root.

## Completed run

Scientific root:

Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5

Presentation root:

Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_deliverables

The run completed 66 calibration fits, 96 expanded fits, 15 objective-diverse
confirmed configurations, and 1,110 pipeline lanes. CUDA projection and MSLN
parity errors were below 7.2e-7.

Raw -> MSICA selected 39/79 known matches at the fixed budget of 58 candidates
per burst, below Raw Direct at 49/79. The global pipeline selector selected
44/79. Within the separately predeclared objective families, full-embedding
CS-Parzen and normalized HSIC each reached 54/79. Treat these as provisional
family-conditioned findings because eight objective/formulation strata were
inspected. The 60/79 protected ceiling was chosen with labels and is diagnostic
only.

## Next decision

For an independent recording, freeze full-embedding CS-Parzen and normalized
HSIC as two confirmation arms with their exact family-specific label-free
context rules. Do not promote the protected ceiling. Improve the global
label-free selector by combining held-out objective gain, seed/map consistency,
and event/quiet contrast before any further label-driven comparison.
