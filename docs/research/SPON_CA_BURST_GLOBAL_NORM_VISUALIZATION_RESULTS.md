# Spon Ca Burst global-normalization visualization results

## Current practical decision

The preferred practical visualization architecture is:

```text
Raw -> MSICA -> MSLN -> GN
```

`GN` is `max(2 ** (alpha * MSLN) - 1, 0)`. MSLN and GN use black at zero
and one fixed movie-wide global scale. The global scale is the preferred
default after direct comparison with framewise, rolling-window, and causal
adaptive alternatives. It preserves brightness comparability across the whole
recording and avoids gain pumping, window-boundary changes, look-ahead, and
adaptive scale lag.

## Experiments completed

### Temperature visualization and benchmark

The expanded sweep evaluated positive MSLN and 15 alpha values from 0.025 to
1.0 under max, mean, top-3 mean, and top-5 mean temporal pooling: 64 lanes in
total. Annotation-free videos are under:

```text
Outputs/HierarchicalParzenICA/
  spon_ca_burst_zero_anchored_temperature_benchmark_v1/
    scientific_audit/4_Vis_Diagnostics/
```

Lower alpha values around 0.1--0.25 remain visually attractive. Alpha 0.5 is
the current benchmark candidate because it participated in the strongest
label-informed lane. Visual preference and detection efficacy remain separate.

Maximum pooling was an invariance control: every monotone temperature arm
produced identical candidate rankings, as expected. At 58 candidates per burst:

- primary mean-family leave-one-burst-out selection: 49/79 known positives;
- broader protected pooling-family selection: 49/79;
- historical Raw Direct anchor: 49/79;
- post-hoc alpha 0.5/top-5 ceiling: 54/79;
- post-hoc alpha 0.4/top-5 ceiling: 54/79.

The protected result tied Raw Direct. The 54/79 result is a label-informed
diagnostic ceiling requiring prospective confirmation. The broad 53/79 top-3
plateau suggests temporal aggregation contributed more than fine alpha tuning.
Unmatched candidates remain unknown because labels are sparse positives.

### Display-scale comparisons

All comparisons used alpha 0.5, zero anchoring, grayscale, and no annotations.

1. **Exact framewise scaling** increased visibility but destroyed temporal
   brightness comparability. Its response maximum ranged from 2.21 to 60.82,
   allowing quiet frames to be amplified by roughly an order of magnitude.
2. **Centered rolling maxima** at 15, 31, and 61 frames reduced some pumping
   but changed abruptly when extrema entered or left a window. They were
   non-causal diagnostics.
3. **Causal exponentially weighted power means** used a 15-frame inferred
   300-ms half-life. Powers 4, 8, and 16 reduced mean absolute log-scale
   movement from 0.159 for framewise scaling to 0.0180, 0.0130, and 0.00884.
   Pixel clipping remained below 0.00013%. Power 8 was the best adaptive
   compromise, but still lacked whole-movie brightness comparability.
4. **Causal attack/release scaling** was more responsive but less smooth than
   the power-mean alternatives.

These results establish that causal adaptive display is feasible if local
visibility becomes necessary. They do not displace global scaling as the
standard scientific visualization.

## Default contract

- Preferred order: `Raw -> MSICA -> MSLN -> GN`.
- Preferred scale: one fixed global movie-wide scale.
- Zero reference: black for positive MSLN and GN.
- Numerical arrays: float32 or higher; global extrema recorded as float64;
  uint8 only at encoding.
- Framewise and adaptive normalization: labeled diagnostics only.
- Scientific arrays and visualization mappings remain separate artifacts.
- Alpha 0.5/top-5 is a prospective candidate, not a validated winner.

## Next defensible experiment

On an independently annotated recording, freeze alpha 0.5/top-5 before label
access and compare it with alpha 0.4/top-5, alpha 1.0/mean, positive
MSLN/top-5, and Raw Direct. Continue producing the standard Expert
Annotations, Model Annotations, Comparison, and annotation-free Visualization
Diagnostics sections. Until then, use the pipeline for visual exploration and
treat detection gains as provisional.
