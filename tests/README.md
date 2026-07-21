# Test Guide

Use the project virtual environment. The default focused path is CPU-only and
must not launch sweeps, train models, or require CUDA.

## Spon Soma-Excitation Focus

~~~bash
.venv-neurobench/bin/python -m pytest \
  tests/test_video_store.py \
  tests/test_cfar_algorithm.py \
  tests/test_chunked_processing.py \
  tests/test_experiment_cli.py \
  tests/test_soma_excitation_preflight.py \
  tests/test_soma_excitation_zones.py \
  tests/test_soma_excitation_detector.py \
  tests/test_soma_excitation_transfer.py \
  tests/test_soma_excitation_runner.py \
  tests/test_cli_main.py
~~~

These tests use tiny synthetic arrays and checkpoints. They protect frame-index
semantics, bounded chunks, quiet-only calibration, dark-core/ring separation,
batch-1 inference, import-order resource limits, live RSS enforcement, and
output collision guards.

## Broader CPU Regression

Run the nearest focused test module first. Before sharing a change, add the
relevant neighboring modules rather than defaulting to an overnight or GPU
workflow. Commands documented in AGENTS.md remain authoritative for stopped
sweep and launch-readiness code.

GPU training, full-video artifact generation, and sweep resume commands are
operator actions, not ordinary unit tests. Never include them in a generic test
helper.
