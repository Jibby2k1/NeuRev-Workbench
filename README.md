# Neuron Detection Grid Search Pipeline

A complete, modular pipeline for **neuron detection** in voltage-imaging videos.
It supports interchangeable **pre-processing filters** (Gamma spatio-temporal and Kalman–MCC background subtraction), **CFAR** detection, **parallel grid search**, and a rich suite of **reports & figures**.

---

## Features

* **Interchangeable pre-processing filters**

  * **Gamma ST filter** (spatio-temporal enhancement)
  * **Kalman–MCC** (robust background estimation via maximum correntropy)
* **Parallel grid search** across filter & detector parameters
* **Evaluation & selection**

  * FROC (TPR vs FPPI), truncated AUC, Youden’s J (balanced operating point)
  * Top-K and balanced model summaries
* **Reports & figures**

  * Per-model detailed reports with diagnostics
  * Event duration histogram, FROC, sensitivity heatmaps
  * PSD comparison, frame-wise power, qualitative montages, error maps
* **Clean, modular codebase** (easy to add new filters or detectors)

---

## Repository Structure

```
.
├── main.py
├── config.py
├── data_loader.py
├── utils.py
├── worker.py
├── core/
│   ├── filters.py
|   ├── detection.py
│   └── pipelines.py
├── evaluation/
│   ├── analysis.py
│   └── metrics.py
├── reporting/
│   ├── generators.py
│   └── plotters.py
├── Inputs/
└── Outputs/
```

---

## Installation

> Requires **Conda** (Anaconda or Miniconda). Python ≥ 3.8. A CUDA GPU is recommended.

1. Create the environment:

```bash
conda env create -f environment.yml
```

2. Activate it:

```bash
conda activate neuron-detection
```

3. (Optional) Update later:

```bash
conda env update -f environment.yml --prune
```

### Example `environment.yml`

> Save as `environment.yml` in the repo root.

```yaml
name: neuron-detection
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - matplotlib
  - scikit-image
  - tifffile
  - tqdm
  - psutil
  - pytorch
  - pytorch-cuda=12.1  # match your local CUDA driver; or remove if CPU-only
  - cudatoolkit=12.1   # optional; Conda-forge uses "cuda-toolkit"
  - pip
  - pip:
      - cupy-cuda12x    # optional, for GPU-accelerated Kalman–MCC (pick the right wheel)
```

**Notes**

* For **CPU-only**: remove the `pytorch-cuda`/`cudatoolkit` lines and `cupy-cuda12x`.
* For **different CUDA versions**, change `*-cuda12x` accordingly.

---

## Inputs

Place files in `Inputs/`:

* `video1_cropped_adj.tif` — your TIFF video (T×H×W)
* `Neuron and Blood Vessel labeled_CL(Video1_Neuron)_cropped_adj.csv` — neuron GT CSV
  Required columns: `ID, Start Frame, End Frame, X, Y`
* (Optional) Vessel GT CSV for FP taxonomy

Update paths in `config.py` if your filenames differ.

---

## Configuration

Edit **`config.py`** to control paths and parameter sweeps.

### CFAR & Shared Settings

```python
SHARED_PARAMS = {
    'distance_tolerance': [6],
    'neighborhood_config': [(3, 1)],
    'cfar_config': [{
        'type': 'local-separate-gamma',
        'gamma_radial_params': (9, 35),
        'kernel_size': (23, 23),
        'eps': 64
    }]
}
TRUNCATION_FPPI_LIMIT = 30.0
TOP_K_MODELS_TO_REPORT = 16
Z_SCORE_SWEEP = np.linspace(0, 2, 256)
```

### Choose Filters to Search

You can search **Gamma**, **Kalman–MCC**, or **both** (combined).
Uncomment / set either or both parameter blocks.

**Gamma example:**

```python
GAMMA_PARAMS = {
    'filter_type': ['gamma'],
    't_decay': [2**(-3) * i for i in range(9)],
    's_decay': [2**(-2) * i for i in range(3)],
}
```

**Kalman–MCC example:**

```python
KALMAN_PARAMS = {
    'filter_type': ['kalman_mcc'],
    'sigma': [2**(-2) * i for i in range(5)],
    'mu':    [2**(-2) * i for i in range(5)],
}
```

> The pipeline treats filters as interchangeable.
> If **both** `GAMMA_PARAMS` and `KALMAN_PARAMS` are defined, the grid search will run **each family separately** and also a **combined comparison** (to see the true overall best), while still reporting each family’s results individually.

---

## Running

```bash
python main.py
```

What happens:

1. Load video & ground truth.
2. Build the parameter grid(s) for the selected filter(s).
3. Run **parallel grid search** (uses `NUM_WORKERS`).
4. Save `Outputs/GridSearch_Full_Report_*/…` with:

   * `all_grid_search_results.csv`
   * `top_models_at_100_tpr.csv`
   * `top_models_balanced.csv`
   * Per-model `Rank_*_Report/` folders
   * Figures (`.png`) and diagnostics (`.tif`, `.csv`)

---

## Outputs & Figures

* **FROC**: `fig_froc_top_k.png`
* **Sensitivity heatmaps**: `sensitivity_*_auc.png`
* **Event duration histogram**: `fig_event_duration_histogram.png`
* **PSD comparison**: `fig_psd_comparison.png`
* **Frame-wise power**: `fig_power_analysis_*.png`
* **Qualitative montage**: `fig_qualitative_montage.png`
* **Error maps**: `fig_fp_density.png`, `fig_fn_per_id.png`
* **Per-model video stack**: `diagnostic_video.tif` (raw | features | z-score | pre/post masks)

---

## Tips & Performance

* **Kalman–MCC** is heavier than Gamma. To speed up:

  * Shrink the Kalman grid (`sigma`, `mu`)
  * Reduce CI bootstraps (see `analysis.calculate_bootstrap_ci`)
  * Limit frames for Kalman visualization: `KALMAN_MCC['max_frames_for_plots']`
  * Prefer GPU with CuPy if available
* Ensure your ground-truth CSV covers valid frame indices.
* If a plot looks off, delete the affected `Outputs/` folder and re-run to regenerate artifacts.

---

## Troubleshooting

* **No models returned**
  Verify GT frames overlap with the video; loosen `distance_tolerance`.
* **CI very slow**
  Reduce bootstrap count and/or sample sizes; disable CI for exploratory runs.
* **Matplotlib missing**
  Recreate the Conda env or `conda install matplotlib`.

---

## License

MIT — do whatever you want, just don’t blame us if it breaks 😊

---

## Citation

If this pipeline helps your work, please cite the repository in your methods or acknowledgements.
