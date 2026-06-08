# Grid Latent Dynamics

This document describes the first model path after template registration and
32x32 grid extraction. The goal is to learn compact grid-frame latents and a
small recurrent next-state predictor while preserving video-level validation.

The model path is:

```text
x_t = 32x32 grid frame
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
  --prediction-horizon-frames 1 \
  --out-dir Outputs/GridModel/dynamics
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

## Train The Grid Autoencoder

```bash
python -m neurobench.cli.main dynamics train-autoencoder \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --latent-dim 32 \
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
  --out-dir Outputs/GridModel/models/latent_rnn_v1
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
