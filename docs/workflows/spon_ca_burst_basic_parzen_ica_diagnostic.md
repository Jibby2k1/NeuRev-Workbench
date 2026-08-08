# Spon Ca Burst basic two-frame CS-Parzen ICA diagnostic

## Status

The completed diagnostic-only package is:

```text
Outputs/PairwiseSeparation/spon_ca_burst_basic_parzen_ica_diagnostic_v1
```

It reuses the immutable fitted arrays from:

```text
Outputs/PairwiseSeparation/spon_ca_burst_pairwise_separation_v1
```

No ICA refit is performed. The package is for method understanding and
interpretation; it does not promote CS-Parzen ICA as validated physical source
separation or replace Raw Direct.

## Primary artifacts

- `METHOD_REFERENCE.md`: complete formula-first method and interpretation note.
- `CHATGPT_HANDOFF.md`: self-contained local-data and preliminary-results context.
- `VIDEO_GUIDE.md`: exact panel, scaling, timing, and overlay definitions.
- `videos/basic_two_frame_cs_parzen_ica_diagnostic.mp4`: 560-frame grayscale review.
- `preliminary_metrics.json`: fitted direction and Parzen-versus-derivative measurements.
- `chatgpt_upload_bundle.zip`: compact upload bundle without the 115 MB MP4.
- `representative_frames/`: quiet and burst frames rendered with the video contract.

The separately validated portable evidence report is:

```text
Outputs/PairwiseSeparation/spon_ca_burst_basic_parzen_ica_report_v1/report.html
```

## Video contract

The MP4 covers UI frames 1800--2359 inclusive. Acquisition was 50 Hz and the
video is encoded at 10 fps, so playback is five times slower than acquisition.
Every panel uses a fixed full-interval display range.

1. Raw observation: `R_t`, original uint16 fluorescence.
2. Causal input: `P_t = EMA_0.4(G_sigma=1px * R_t)`.
3. Fixed derivative: `D_t = P_t - P_(t-1)`.
4. CS-Parzen activity:
   `Y_t = s e_k^T W Q ([P_(t-1),P_t]^T - mu)`.
5. Positive quiet score:
   `Z_t+ = max(0,(Y_t-med_Q)/max(1.4826 MAD_Q,floor))`.
6. Non-derivative residual:
   `E_t = Y_t-beta D_t`, with `beta` fitted on quiet frames.

Raw/preprocessed panels use black for low and white for high. Signed panels use
black for negative, mid-gray for zero, and white for positive. Black-backed
white rings appear only during their annotated burst interval. They are sparse
known positives; unmarked pixels remain unknown.

## Reproduction

The generator refuses an existing final or `.partial` root:

```bash
.venv-neurobench/bin/python -m \
  neurobench.experiments.pairwise_separation.basic_parzen_diagnostic \
  --completed-run Outputs/PairwiseSeparation/spon_ca_burst_pairwise_separation_v1 \
  --raw-npy Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy \
  --labels-tsv "Inputs/Spon Ca Burst/labels/labels_normalized.tsv" \
  --output-dir Outputs/PairwiseSeparation/<new_diagnostic_root> \
  --fps 10
```

The portable report artifact is generated from a completed diagnostic package:

```bash
.venv-neurobench/bin/python -m neurobench.reports.basic_parzen_ica \
  --diagnostic-root Outputs/PairwiseSeparation/<completed_diagnostic_root> \
  --output-dir Outputs/PairwiseSeparation/<new_report_root>
```

The report HTML must then be built with the pinned Data Analytics portable
artifact builder. The checked-in report generator deliberately keeps the HTML
surface compact because the current reader overflows when long narrative blocks
activate vertical scrolling; the complete narrative remains in
`METHOD_REFERENCE.md`.

## Measured interpretation

- Normalized learned axis: `[-0.707394214, +0.706819231]`.
- Absolute cosine to the derivative direction: `0.999999917`.
- Sampled activity correlation with fixed differencing: `0.997828107`.
- Quiet-aligned non-derivative residual normalized RMS: `0.0689458033`.
- CS-Parzen known-label mean recall: `0.1333`, 10/79 matches, 24 candidates.
- Raw Direct reference: `0.6056`, 49/79 matches, 232 candidates.

Candidate counts are not precision because labels are not exhaustive. The
residual retains visible anatomical structure and must not be called measurement
noise.
