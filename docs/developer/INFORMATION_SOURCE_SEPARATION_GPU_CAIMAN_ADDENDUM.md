# Information Source Separation: GPU and CaImAn Addendum

## Frozen execution decision

The generated screen uses the GPU only for the quadratic normalized-HSIC
dependence objective. PCA whitening remains the NumPy reference and the PCA,
multi-lag SOBI, and kNN-MI configurations remain CPU references. Every fit
records `execution_backend`; summaries must not describe the whole panel as a
GPU benchmark.

CUDA HSIC uses PyTorch float64 on `cuda:0`. A read-only preflight compares the
CPU and CUDA normalized-HSIC definitions and requires absolute error at most
`1e-10`. The output root is new and resumable:

```text
Outputs/InformationSourceSeparation/generated_screen_gpu_v1
```

The selected screen is 14 fixtures by 48 configurations, or 672 fits. It does
not authorize the later 1,365-fit generated confirmation, semi-synthetic Spon
injection, CNMF fitting, or full Spon benchmark.

## CaImAn reference environment

CaImAn is isolated from `.venv-neurobench` because its compiled imaging stack
and solver constraints are independent of the maintained workbench runtime.
The installed contract is:

- Miniforge `26.3.2-3`, verified against the release SHA-256;
- Python `3.11`;
- CaImAn `1.13.1` from conda-forge;
- environment prefix
  `/home/jibby2k1/.local/share/neurobench-caiman-1.13.1`;
- explicit solved-package fingerprint
  `90586c2e03a7dec5c73cb7856f2edb82cd619b5f692c444b33ea291ab7a85b2d`.

The human-maintained recreation manifest is
`environment.caiman-linux-64.yml`. The adapter audits the isolated interpreter,
imports `caiman.source_extraction.cnmf.cnmf.CNMF`, and requires the exact
manifest version. Ordinary NMF is never substituted.

Set these variables for CaImAn commands to prevent nested BLAS oversubscription:

```bash
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

CaImAn's own documentation describes it as primarily multi-CPU software with
limited experimental GPU use. Therefore a future CNMF reference fit must be
budgeted and interpreted separately from the CUDA HSIC screen.

## Commands

Read-only CUDA preflight:

```bash
.venv-neurobench/bin/python -m \
  neurobench.experiments.information_source_separation.gpu_cli preflight \
  --config examples/spon_ca_burst_information_source_separation_gpu_v1.example.json \
  --output-dir Outputs/InformationSourceSeparation/generated_screen_gpu_v1
```

An interrupted run may be resumed only after auditing the partial root and
rerunning preflight against a distinct proposed path or by direct inspection of
the existing partial contract. Completed roots are never overwritten.
