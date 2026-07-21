# Grid Latent Dynamics

This document describes the model path after template registration and grid
state extraction. The current high-resolution experiment uses 128x128
max-pooled grid states from the cropped 512x512 videos. The goal is to learn
compact grid-frame latents and a small recurrent next-state predictor while
preserving video-level validation.

The model path is:

```text
x_t = 128x128 max-pooled grid frame
z_t = CNN encoder(x_t)
recon_t = CNN decoder(z_t)
z_hat_next = GRU(z_window)
x_hat_next = CNN decoder(z_hat_next)
```

## Build The Dynamics Dataset

```bash
python -m neurobench.cli.main dynamics build-dataset \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --grid-states-dir Outputs/GridModel/grid_states \
  --split-unit video \
  --split-method stratified_by_label \
  --window-frames 8 \
  --prediction-horizon-frames 2 \
  --temporal-stride-frames 1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2
```

Outputs:

- `dynamics_dataset.json`
- `dynamics_arrays.npz`
- `split_manifest.json`

Windows are created within a video only. Train, validation, and test assignments
are video ids, never random frames.

## Evaluate Baselines

Run baselines before interpreting a learned predictor:

```bash
python -m neurobench.cli.main dynamics evaluate-baselines \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --out Outputs/GridModel/dynamics/baseline_metrics.json
```

The MVP baselines are persistence and moving average. The latent RNN report must
include persistence comparison.

## Timing Defaults

For 50 Hz calcium videos, each frame is 20 ms. The short-horizon experiment
builds two datasets from the same max-pooled grid states:

- `prediction_horizon_frames=2`, about 40 ms.
- `prediction_horizon_frames=5`, about 100 ms.

These cover the plausible 30 Hz and 10 Hz calcium-timescale interpretations
without forcing one biological assumption into the code. The dataset JSON records
`source_frame_rate_hz`, `effective_frame_rate_hz`, `window_sec`, and
`prediction_horizon_sec` under `windowing`.

## Train The Grid Autoencoder

```bash
python -m neurobench.cli.main dynamics train-autoencoder \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --latent-dim 64 \
  --epochs 10 \
  --batch-size 32 \
  --out-dir Outputs/GridModel/models/autoencoder_v1
```

Outputs:

- `autoencoder_run.json`
- `autoencoder_checkpoint.pt`
- `autoencoder_metrics.json`
- `reconstruction_examples.json`
- `reconstruction_examples.png`
- `latent_codes.npz`

The current architecture is intentionally small and CPU-safe:

```text
Conv/ReLU/Pool -> Conv/ReLU/Pool -> latent vector -> upsample decoder
```

## Train The Latent GRU Predictor

```bash
python -m neurobench.cli.main dynamics train-latent-rnn \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/models/autoencoder_v1/autoencoder_run.json \
  --window-frames 8 \
  --hidden-dim 64 \
  --epochs 10 \
  --batch-size 32 \
  --prediction-target delta \
  --out-dir Outputs/GridModel/models/latent_rnn_h2_delta_v1
```

Outputs:

- `latent_rnn_run.json`
- `latent_rnn_checkpoint.pt`
- `latent_rnn_metrics.json`
- `baseline_metrics.json`
- `prediction_examples.json`
- `prediction_examples.png`

The learned model should not be described as useful unless it beats persistence
on held-out videos or the report clearly states the subset and limitation.

## Why The GRU/RNN Baseline

A temporal CNN is a reasonable ablation because it can learn fixed-window
temporal filters from stacked frames. The GRU is the primary baseline because the
calcium movie is an ordered state sequence: at 50 Hz, the model sees frames every
20 ms, and the prediction depends on recent temporal context rather than on an
independent image. A GRU keeps an explicit hidden state over the latent-code
history, which is easier to justify for short calcium transients and for future
online use than treating all past frames only as extra image channels.

For the short 2-frame and 5-frame horizons, `prediction_target=delta` is the
preferred recurrent objective. It asks the GRU to predict the next latent change
from the last observed state, while decoded outputs are still compared against
the absolute target frame and against persistence.

## Sweep Hyperparameters

Use the capped sweep command for sequential AE + latent-GRU searches. It ranks
latent GRU candidates by standardized next-code MSE first and reports decoded
next-grid MSE as evaluation-only context.

```bash
python -m neurobench.cli.main dynamics sweep-latent-dynamics \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --device auto \
  --latent-dims 16,32,64 \
  --autoencoder-epochs 10,25 \
  --autoencoder-learning-rates 0.001,0.0003 \
  --rnn-hidden-dims 32,64,128 \
  --rnn-epochs 10,25 \
  --rnn-learning-rates 0.001,0.0003 \
  --max-autoencoders 6 \
  --max-rnn-runs 24 \
  --out-dir Outputs/GridModel/sweeps/latent_dynamics_v1
```

Outputs:

- `sweep_summary.json`
- `sweep_results.tsv`
- per-candidate `autoencoder_run.json` and `latent_rnn_run.json`

The command is deliberately sequential and capped. To sweep `window_frames`,
rebuild one dynamics dataset per window length; the latent RNN uses the windows
already stored in the provided dataset.

For the 128x128 max-pooled workflow, use the bounded architecture profile. With
the default two horizons (`w8_s1_h2`, `w8_s1_h5`) and seeds `7,13`, it creates
972 experiments across array baselines, linear latent baselines, latent GRUs,
latent Transformers, ConvGRU/ConvLSTM pixel models, and temporal CNN pixel
models. Experiment IDs use the `g128_` prefix and each config includes a compact
`hyperparameter_summary` for the comparison UI.

```bash
python -m neurobench.dynamics.overnight_sweep \
  --profile grid128_sequence_1day \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --device cuda \
  --epochs 50 \
  --batch-size 64 \
  --seeds 7,13
```

Build the static comparison dashboard after or during the run:

```bash
python -m neurobench.dynamics.comparison \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1
```

## Train The Latent Classifier

The classifier uses video-level summaries of encoder latent codes:

```text
summary = concat(mean_t(z_t), std_t(z_t))
```

Run:

```bash
python -m neurobench.cli.main dynamics train-latent-classifier \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/models/autoencoder_v1/autoencoder_run.json \
  --labels-from manifest \
  --split-unit video \
  --evaluation stratified_kfold \
  --out-dir Outputs/GridModel/classifier/latent_classifier_v1
```

Outputs:

- `latent_classifier_run.json`
- `per_video_predictions.tsv`
- `confusion_matrix.png`
- `latent_embedding_2d.png`

The classifier labels come from filenames only: `neutral`, `left`, and `right`.
For balanced three-class data, chance accuracy is about 33.3 percent.

## Example Pipeline

The dynamics model stages are represented in:

```text
examples/grid_latent_dynamics_pipeline.example.json
examples/template_grid_128x128_pipeline.example.json
```

Keep example epochs tiny for CI and CPU smoke tests. Real experiments should
record the dataset, reference template, split ids, random seed, device, and
baseline comparison with each model run.

## Guardrails

- Keep split unit as `video`.
- Do not include stimulation/control inputs in current runs.
- Do not train transformer models before the GRU baseline, persistence
  comparison, and video-split validation are stable.
- Do not commit raw real videos or large checkpoints unless the team explicitly
  decides to version them.
